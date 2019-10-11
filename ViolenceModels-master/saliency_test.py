import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

from saliency_model import *
from util import createDataset, getViolenceData
import random
from ViolenceDatasetV2 import ViolenceDatasetVideos
from operator import itemgetter
from transforms import createTransforms
from loss import Loss

num_workers = 1
batch_size = 3

def imshow(img):
    img = img / 2 + 0.5    
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def subplot(img1, img2, img3):
    fig2 = plt.figure(figsize=(12,12))
    img1 = img1 / 2 + 0.5    
    npimg1 = img1.numpy()
    plt.subplot(311)
    plt.imshow(np.transpose(npimg1, (1, 2, 0)))

    img2 = img2 / 2 + 0.5    
    npimg2 = img2.numpy()
    plt.subplot(312)
    plt.imshow(np.transpose(npimg2, (1, 2, 0)))

    img3 = img3 / 2 + 0.5    
    npimg3 = img3.numpy()
    plt.subplot(313)
    plt.imshow(np.transpose(npimg3, (1, 2, 0)))

    plt.show()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset = torchvision.datasets.CIFAR10(root='data/', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=num_workers)

# testset = torchvision.datasets.CIFAR10(root='data/', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=num_workers)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

hockey_path_violence = "/media/david/datos/Violence DATA/HockeyFights/frames/violence"
hockey_path_noviolence = "/media/david/datos/Violence DATA/HockeyFights/frames/nonviolence"
datasetAll, labelsAll, numFramesAll = getViolenceData(hockey_path_violence) #ordered
# combined = list(zip(datasetAll, labelsAll, numFramesAll))
# random.shuffle(combined)
# datasetAll[:], labelsAll[:], numFramesAll[:] = zip(*combined) 
print(len(datasetAll), len(labelsAll), len(numFramesAll))

interval_duration = 0
avgmaxDuration = 0
numDiPerVideos = 1
input_size = 224
data_transforms = createTransforms(input_size)
dataset_source = 'frames'
debugg_mode = False
batch_size = 1
num_workers = 1
num_epochs = 3
num_classes = 2

image_datasets = {
    "train": ViolenceDatasetVideos( dataset=datasetAll, labels=labelsAll, spatial_transform=data_transforms["train"], source=dataset_source,
        interval_duration=interval_duration,difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, ),
    # "val": ViolenceDatasetVideos( dataset=dataset_test, labels=dataset_test_labels, spatial_transform=data_transforms["val"], source=dataset_source,
    #     interval_duration=interval_duration, difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, )
}
dataloaders_dict = {
    "train": torch.utils.data.DataLoader( image_datasets["train"], batch_size=batch_size, shuffle=False, num_workers=num_workers, ),
    # "val": torch.utils.data.DataLoader( image_datasets["val"], batch_size=batch_size, shuffle=True, num_workers=num_workers, ),
}
net = saliency_model(num_classes=num_classes)
net = net.cuda()

net = torch.load('/media/david/datos/Violence DATA/HockeyFights/checkpoints/saliency_model.tar')
padding = 20
# plt.figure(1)
for i, data in enumerate(dataloaders_dict['train'], 0):
    print('video: ',i,' ',datasetAll[i])
    inputs_r, labels = data #dataset load [bs,ndi,c,w,h]
    # print('dataset element: ',inputs_r.shape)
    inputs_r = inputs_r.permute(1, 0, 2, 3, 4)
    inputs = torch.squeeze(inputs_r, 0) #get one di [bs,c,w,h]
    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    masks, _ = net(inputs, labels)
    #Original Image
    di_images = torchvision.utils.make_grid(inputs.cpu().data, padding=padding)
    #Mask
    mask = torchvision.utils.make_grid(masks.cpu().data, padding=padding)
    #Image Segmented
    segmented = torchvision.utils.make_grid((inputs*masks).cpu().data, padding=padding)
    subplot(di_images, mask, segmented)
    
    
   