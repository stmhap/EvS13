from model import Model
from pytorch_lightning import Trainer

model = Model(batch_size=8, enable_gc='batch')
trainer = Trainer(max_epochs=40)

trainer.fit(model)
