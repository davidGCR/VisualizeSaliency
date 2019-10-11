import torch
import torchvision

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models
import torch
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable

# from tensorboardcolab import TensorBoardColab
import time
from torch.optim import lr_scheduler
import numpy as np

import sys

sys.path.insert(1, "/media/david/datos/PAPERS-SOURCE_CODE/MyCode")
from AlexNet import *
from ViolenceDatasetV2 import *
from trainer import *
from kfolds import *
from operator import itemgetter
import random
from initializeModel import *
from util import *
from verifyParameters import *

from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler
from skorch.callbacks import Checkpoint
from sklearn.model_selection import cross_val_score
from skorch.callbacks import Freezer


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_size = 224

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

# tb = TensorBoardColab()

train_lost = []
train_acc = []
test_lost = []
test_acc = []

foldidx = 0
best_acc_test = 0.0
avgmaxDuration = 1.66


modelType = "alexnetv1"
interval_duration = 0.3
numDiPerVideos = 3
dataset_source = "frames"
debugg_mode = False
num_workers = 4
batch_size = 64
num_epochs = 50
feature_extract = True
joinType = "tempMaxPool"
# path_models = '/media/david/datos/Violence DATA/violentflows/Models/'
# path_results = '/media/david/datos/Violence DATA/violentflows/Results/'+dataset_source
# gpath = '/media/david/datos/Violence DATA/violentflows/movies Frames'

debugg_mode = False

path_models = "/media/david/datos/Violence DATA/HockeyFights/Models"
path_results = "/media/david/datos/Violence DATA/HockeyFights/Results/" + dataset_source

# Create dataset HockeyFights
path_violence = "/media/david/datos/Violence DATA/HockeyFights/frames/violence"
path_noviolence = "/media/david/datos/Violence DATA/HockeyFights/frames/nonviolence"
datasetAll, labelsAll, numFramesAll = createDataset(path_violence, path_noviolence)
combined = list(zip(datasetAll, labelsAll, numFramesAll))
random.shuffle(combined)
datasetAll[:], labelsAll[:], numFramesAll[:] = zip(*combined)
print(len(datasetAll), len(labelsAll), len(numFramesAll))

image_datasets = {
    "train": ViolenceDatasetVideos(
        dataset=datasetAll,
        labels=labelsAll,
        spatial_transform=data_transforms["train"],
        source=dataset_source,
        interval_duration=interval_duration,
        difference=3,
        maxDuration=avgmaxDuration,
        nDynamicImages=numDiPerVideos,
        debugg_mode=debugg_mode,
    ),
    "val": ViolenceDatasetVideos(
        dataset=datasetAll,
        labels=labelsAll,
        spatial_transform=data_transforms["val"],
        source=dataset_source,
        interval_duration=interval_duration,
        difference=3,
        maxDuration=avgmaxDuration,
        nDynamicImages=numDiPerVideos,
        debugg_mode=debugg_mode,
    ),
}
dataloaders_dict = {
    "train": torch.utils.data.DataLoader(
        image_datasets["train"],
        batch_size=1000,
        shuffle=True,
        num_workers=num_workers,
    ),
    # "val": torch.utils.data.DataLoader(
    #     image_datasets["val"],
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    # ),
}

X_train = None
y_train = None
for inputs, labels in dataloaders_dict["train"]:
    X_train = inputs
    y_train = labels
X_train = X_train.numpy()
y_train = y_train.numpy()
# print('X_train: ', type(X_train), X_train.size())
# print('y_train: ',type(y_train),y_train.size())

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()
lrscheduler = LRScheduler(policy="StepLR", step_size=7, gamma=0.1)
checkpoint = Checkpoint(f_params="best_model.pt", monitor="valid_acc_best")
freezer = Freezer(lambda x: not x.startswith("model.linear"))
# params_to_update = verifiParametersToTrain(model)
# Observe that all parameters are being optimized
optimizer = optim.SGD
net = NeuralNetClassifier(
    ViolenceModelAlexNetV1,
    criterion=criterion,
    lr=0.001,
    batch_size=64,
    max_epochs=15,
    module__model_name=modelType,
    module__num_classes=2,
    module__feature_extract=feature_extract,
    module__numDiPerVideos=numDiPerVideos,
    module__joinType=joinType,
    module__use_pretrained=True,
    optimizer=optimizer,
    optimizer__momentum=0.9,
    iterator_train__shuffle=True,
    iterator_train__num_workers=4,
    iterator_valid__shuffle=True,
    iterator_valid__num_workers=4,
    train_split=None,
    callbacks=[lrscheduler],
    device="cuda",  # comment to train on cpu
)

# scores = cross_val_score(net, image_datasets['train'], np.array(labelsAll), cv=5)
# print(scores)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from sklearn.model_selection import cross_val_predict

y_pred = cross_val_predict(net, image_datasets["train"], y_train, cv=5)
print(y_pred)
