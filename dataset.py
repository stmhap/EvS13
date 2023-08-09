"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import config
import numpy as np
import os
import pandas as pd
import torch
from utils import ResizeDataLoader, show_transform
import random
import itertools

from PIL import Image, ImageFile
from torch.utils.data import Dataset

from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image,
    xywhn2xyxy,
    xyxy2xywhn
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=config.IMAGE_SIZE,
        S=config.S,
        transform=None,
        mosaic=0.75,
        targets=True
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.mosaic = mosaic
        self.targets = targets
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def load_image(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        img = np.array(Image.open(img_path).convert("RGB"))

        return img, bboxes

    def load_mosaic(self, index, p=0.5):
        if random.random() >= p:
            return self.load_image(index)

        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4 = []
        s = self.image_size
        mosaic_border = [s // 2, s // 2]
        yc, xc = (int(random.uniform(x, 2 * s - x)) for x in mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(range(len(self)), k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, bboxes = self.load_image(index)

            h, w = img.shape[0], img.shape[1]
            labels = np.array(bboxes)

            # place img in img4
            if i == 0:  # top left
                # img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                img4 = np.full((s * 2, s * 2, img.shape[2]), np.array(config.mean) * 255, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            if labels.size:
                labels[:, :-1] = xywhn2xyxy(labels[:, :-1], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            labels4.append(labels)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, :-1],):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate
        labels4[:, :-1] = xyxy2xywhn(labels4[:, :-1], 2 * s, 2 * s)
        labels4[:, :-1] = np.clip(labels4[:, :-1], 0, 1)
        labels4 = labels4[labels4[:, 2] > 0]
        labels4 = labels4[labels4[:, 3] > 0]
        return img4, labels4

    def __getitem__(self, index):
        image, bboxes = self.load_mosaic(index, p=self.mosaic)

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        if not self.targets:
            return image, bboxes

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)


def test():
    anchors = config.ANCHORS

    transform = config.train_transforms

    dataset = YOLODataset(
        "../../input/pascal-voc-dataset-used-in-yolov3-video/PASCAL_VOC/train.csv",
        "../../input/pascal-voc-dataset-used-in-yolov3-video/PASCAL_VOC/images/",
        "../../input/pascal-voc-dataset-used-in-yolov3-video/PASCAL_VOC/labels/",
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    scaled_anchors = config.SCALED_ANCHORS
    loader = ResizeDataLoader(dataset=dataset, batch_size=2, shuffle=True, resolutions=config.MULTIRES,
                              cum_weights=config.CUM_PROBS)
    for x, y in itertools.islice(loader, 10):
        boxes = []
        # print(type(y), len(y), y[0].shape, y[1].shape, y[2].shape)
        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            # print(anchor.shape)
            # print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        # print(type(boxes), len(boxes))
        plot_image(show_transform(x[0]).to("cpu"), boxes)


if __name__ == "__main__":
    test()
