import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from scipy import misc
# from model import saliency_model
# from resnet import resnet
# from loss import Loss
from saliency_model import *
from util import createDataset
import random
from tqdm import tqdm
from ViolenceDatasetV2 import ViolenceDatasetVideos
from operator import itemgetter
from transforms import createTransforms
from loss import Loss


def save_checkpoint(state, filename='sal.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(net,optimizer,filename='small.pth.tar'):
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return net,optimizer

def cifar10():
    
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='data/', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='data/', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    return trainloader,testloader,classes

hockey_path_violence = "/media/david/datos/Violence DATA/HockeyFights/frames/violence"
hockey_path_noviolence = "/media/david/datos/Violence DATA/HockeyFights/frames/nonviolence"
datasetAll, labelsAll, numFramesAll = createDataset(hockey_path_violence, hockey_path_noviolence) #ordered
combined = list(zip(datasetAll, labelsAll, numFramesAll))
random.shuffle(combined)
datasetAll[:], labelsAll[:], numFramesAll[:] = zip(*combined) 
print(len(datasetAll), len(labelsAll), len(numFramesAll))

# dataset_train = list(itemgetter(*train_idx)(datasetAll))
# dataset_train_labels = list(itemgetter(*train_idx)(labelsAll))
# dataset_test = list(itemgetter(*test_idx)(datasetAll))
# dataset_test_labels = list(itemgetter(*test_idx)(labelsAll))

interval_duration = 0
avgmaxDuration = 0
numDiPerVideos = 1
input_size = 224
data_transforms = createTransforms(input_size)
dataset_source = 'frames'
debugg_mode = False
batch_size = 4
num_workers = 4
num_epochs = 3
num_classes = 2

image_datasets = {
    "train": ViolenceDatasetVideos( dataset=datasetAll, labels=labelsAll, spatial_transform=data_transforms["train"], source=dataset_source,
        interval_duration=interval_duration,difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, ),
    # "val": ViolenceDatasetVideos( dataset=dataset_test, labels=dataset_test_labels, spatial_transform=data_transforms["val"], source=dataset_source,
    #     interval_duration=interval_duration, difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, )
}
dataloaders_dict = {
    "train": torch.utils.data.DataLoader( image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers, ),
    # "val": torch.utils.data.DataLoader( image_datasets["val"], batch_size=batch_size, shuffle=True, num_workers=num_workers, ),
}

def train():
    # trainloader,testloader,classes = cifar10()
    net = saliency_model(num_classes=num_classes)
    net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    # black_box_func = resnet(pretrained=True)
    black_box_func = torch.load('/media/david/datos/Violence DATA/HockeyFights/checkpoints/resnet18-frames-Finetuned:False-3di-tempMaxPool-OnPlateau.tar')
    black_box_func = black_box_func.cuda()
    loss_func = Loss(num_classes=num_classes)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_corrects = 0.0
        
        for i, data in tqdm(enumerate(dataloaders_dict['train'], 0)):
            # get the inputs
            inputs_r, labels = data #dataset load [bs,ndi,c,w,h]
            # print('dataset element: ',inputs_r.shape)
            inputs_r = inputs_r.permute(1, 0, 2, 3, 4)
            inputs = torch.squeeze(inputs_r, 0) #get one di [bs,c,w,h]
            # print('inputs shape:',inputs.shape)
            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            mask, out = net(inputs, labels)
            # print('mask shape:', mask.shape)
            # print('inputs shape:',inputs.shape)
            # print('labels shape:',labels.shape)

            # inputs_r = Variable(inputs_r.cuda())
            loss = loss_func.get(mask,inputs,labels,black_box_func)
            # running_loss += loss.data[0]
            running_loss += loss.item()

            if(i%10 == 0):
                print('Epoch = %f , Loss = %f '%(epoch+1 , running_loss/(batch_size*(i+1))) )
        
            loss.backward()
            optimizer.step()
        
        save_checkpoint(net,'/media/david/datos/Violence DATA/HockeyFights/checkpoints/saliency_model.tar')

train()