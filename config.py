import os
import cv2
import torch
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Device Selected:", device)
    return device


DATASET = 'PASCAL_VOC'
DEVICE = get_device()
ACTIVATION = 'relu'
seed_everything()  # If you want deterministic behavior
NUM_WORKERS = min(os.cpu_count(), 4)
BATCH_SIZE = 32
IMAGE_SIZE = 416
MULTIRES = [416, 832]
CUM_PROBS = [4, 4]
MAX_IMAGE_SIZE = MULTIRES[-1]
NUM_CLASSES = 20
LEARNING_RATE = 4*3e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 40
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]

SCALED_ANCHORS = (
    torch.tensor(ANCHORS) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

scale = 1.1

train_transforms = A.Compose(
    [
        A.Posterize(p=0.1),
        A.CLAHE(p=0.1),
        A.Normalize(mean=mean, std=std),
        A.LongestMaxSize(max_size=int(MAX_IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(MAX_IMAGE_SIZE * scale),
            min_width=int(MAX_IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        # A.Rotate(limit=10),
        A.RandomCrop(width=MAX_IMAGE_SIZE, height=MAX_IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.ShiftScaleRotate(rotate_limit=20, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)

test_transforms = A.Compose(
    [
        A.Normalize(mean=mean, std=std),
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0
        ),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]
