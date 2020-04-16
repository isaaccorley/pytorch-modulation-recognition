import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from poutyne.framework import Model

from data import RadioML2016
from models import VT_CNN2, MRResNet


N_CLASSES = 10
EPOCHS = 50
BATCH_SIZE = 512
SPLIT = 0.8
DROPOUT = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
print("Loading Model")

net = VT_CNN2(
    n_classes=N_CLASSES,
    dropout=DROPOUT,
)

"""
net = MRResNet(
    n_channels=2,
    n_classes=N_CLASSES,
    n_res_blocks=8,
    n_filters=32
)
"""

# Load dataset
dataset = RadioML2016()

# Split into train/val sets
total = len(dataset)
lengths = [int(len(dataset)*SPLIT)]
lengths.append(total - lengths[0])
print("Splitting into {} train and {} val".format(lengths[0], lengths[1]))
train_set, val_set = random_split(dataset, lengths)

# Setup dataloaders
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE)

model = Model(
    network=net,
    optimizer="Adam",
    loss_function=nn.CrossEntropyLoss(),
    batch_metrics=["acc"]
)
model.cuda()
model.fit_generator(
    train_dataloader,
    val_dataloader,
    epochs=EPOCHS
)

if not os.path.exists("models"):
    os.mkdir("models")

torch.save(net.state_dict(), "models/vt_cnn2.pt")