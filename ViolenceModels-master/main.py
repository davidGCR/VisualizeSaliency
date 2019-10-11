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
import argparse
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
from transforms import *



def init(dataset, vif_path, hockey_path_violence, hockey_path_noviolence, path_results, path_checkpoints, modelType, ndis, num_workers, data_transforms, dataset_source, interval_duration, avgmaxDuration,
    batch_size, num_epochs, feature_extract, joinType, scheduler_type, device, criterion, folds_number, debugg_mode = False):
    
    for numDiPerVideos in ndis:
        train_lost = []
        train_acc = []
        test_lost = []
        test_acc = []
        if dataset == 'hockey':
            datasetAll, labelsAll, numFramesAll = createDataset(hockey_path_violence, hockey_path_noviolence) #ordered
            combined = list(zip(datasetAll, labelsAll, numFramesAll))
            random.shuffle(combined)
            datasetAll[:], labelsAll[:], numFramesAll[:] = zip(*combined) 
            print(len(datasetAll), len(labelsAll), len(numFramesAll))
            train_idx, test_idx = None, None
        # elif dataset == 'violentflows':

        print("CONFIGURATION: ", "modelType:", modelType, ", numDiPerVideos:", numDiPerVideos,
                ", dataset_source:", dataset_source, ", batch_size:", batch_size, ", num_epochs:",
                num_epochs, ", feature_extract:", feature_extract, ", joinType:", joinType, ", scheduler_type: ", scheduler_type, )

        
        fold = 0
        for train_idx, test_idx in k_folds(n_splits=folds_number, subjects=len(datasetAll)):
        # for dataset_train, dataset_train_labels,dataset_test,dataset_test_labels   in k_folds_from_folders(vif_path, 5):
            fold = fold + 1
            print("**************** Fold: ", fold)
            if dataset == 'hockey':
                print('fold: ',len(train_idx),len(test_idx))
                dataset_train = list(itemgetter(*train_idx)(datasetAll))
                dataset_train_labels = list(itemgetter(*train_idx)(labelsAll))
                dataset_test = list(itemgetter(*test_idx)(datasetAll))
                dataset_test_labels = list(itemgetter(*test_idx)(labelsAll))

            image_datasets = {
                "train": ViolenceDatasetVideos( dataset=dataset_train, labels=dataset_train_labels, spatial_transform=data_transforms["train"], source=dataset_source,
                    interval_duration=interval_duration,difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, ),
                "test": ViolenceDatasetVideos( dataset=dataset_test, labels=dataset_test_labels, spatial_transform=data_transforms["test"], source=dataset_source,
                    interval_duration=interval_duration, difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, )
            }
            dataloaders_dict = {
                "train": torch.utils.data.DataLoader( image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers, ),
                "test": torch.utils.data.DataLoader( image_datasets["test"], batch_size=batch_size, shuffle=True, num_workers=num_workers, ),
            }
            model = None
            model, input_size = initialize_model( model_name=modelType, num_classes=2, feature_extract=feature_extract, numDiPerVideos=numDiPerVideos, joinType=joinType, use_pretrained=True)
            model.to(device)
            params_to_update = verifiParametersToTrain(model)
            # Observe that all parameters are being optimized
            optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
            # Decay LR by a factor of 0.1 every 7 epochs
            if scheduler_type == "StepLR":
                exp_lr_scheduler = lr_scheduler.StepLR( optimizer, step_size=7, gamma=0.1 )
            elif scheduler_type == "OnPlateau":
                exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, patience=5, verbose=True )
            ### trainer
            model_name = get_model_name(modelType, scheduler_type, numDiPerVideos, dataset_source, feature_extract, joinType)
            if folds_number == 1:
                trainer = Trainer(model, dataloaders_dict, criterion, optimizer, exp_lr_scheduler, device, num_epochs, os.path.join(path_checkpoints, model_name))
            else:
                trainer = Trainer( model, dataloaders_dict, criterion, optimizer, exp_lr_scheduler, device, num_epochs, None)

            for epoch in range(1, num_epochs + 1):
                print("----- Epoch {}/{}".format(epoch, num_epochs))
                # Train and evaluate
                epoch_loss_train, epoch_acc_train = trainer.train_epoch(epoch)
                epoch_loss_test, epoch_acc_test = trainer.test_epoch(epoch)
                exp_lr_scheduler.step(epoch_loss_test)
                train_lost.append(epoch_loss_train)
                train_acc.append(epoch_acc_train)
                test_lost.append(epoch_loss_test)
                test_acc.append(epoch_acc_test)
            #   tb.save_value("trainLoss", "train_loss", foldidx*num_epochs + epoch, epoch_loss_train)
            #   tb.save_value("trainAcc", "train_acc", foldidx*num_epochs + epoch, epoch_acc_train)
            #   tb.save_value("testLoss", "test_loss", foldidx*num_epochs + epoch, epoch_loss_test)
            #   tb.save_value("testAcc", "test_acc", foldidx*num_epochs + epoch, epoch_acc_test)

            #   tb.flush_line('train_loss')
            #   tb.flush_line('train_acc')
            #   tb.flush_line('test_loss')
            #   tb.flush_line('test_acc')

            #     filepath = path_models+str(modelType)+'('+str(numDiPerVideos)+'di)-fold-'+str(foldidx)+'.pt'
            #     torch.save({
            #         'kfold': foldidx,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict()
            #         }, filepath)
        print("saving loss and acc history...")
        saveList(path_results, modelType, scheduler_type, "train_lost", numDiPerVideos, dataset_source, feature_extract, joinType, train_lost,)
        saveList(path_results, modelType, scheduler_type,"train_acc",numDiPerVideos, dataset_source, feature_extract, joinType, train_acc, )
        saveList( path_results, modelType, scheduler_type, "test_lost", numDiPerVideos, dataset_source, feature_extract, joinType, test_lost, )
        saveList( path_results, modelType, scheduler_type, "test_acc", numDiPerVideos, dataset_source, feature_extract, joinType, test_acc, )

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str)
    parser.add_argument("--vif_path",type=str,default="/media/david/datos/Violence DATA/violentflows/movies Frames",help="Directory for Violent Flows dataset")
    parser.add_argument("--path_violence",type=str,default="/media/david/datos/Violence DATA/HockeyFights/frames/violence",help="Directory containing violence videos")
    parser.add_argument("--path_noviolence",type=str,default="/media/david/datos/Violence DATA/HockeyFights/frames/nonviolence",help="Directory containing non violence videos")
    parser.add_argument("--path_results",type=str,default="/media/david/datos/Violence DATA/HockeyFights/Results",help="Directory containing results")
    parser.add_argument("--model_type",type=str,default="alexnet",help="model")
    parser.add_argument("--num_epochs",type=int,default=30)
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--feature_extract",type=bool,default=True,help="to fine tunning")
    parser.add_argument("--scheduler_type",type=str,default="OnPlateau",help="learning rate scheduler")
    parser.add_argument("--debugg_mode", type=bool, default=False, help="show prints")
    parser.add_argument("--ndis", nargs='+', type=int, help="num dyn imgs")
    parser.add_argument("--join_type", type=str, default="tempMaxPool", help="show prints")
    parser.add_argument("--checkpoint_path", type=str, default='/media/david/datos/Violence DATA/HockeyFights/checkpoints')
    parser.add_argument("--folds_number", type=int, default=5)

    args = parser.parse_args()

    path_models = "/media/david/datos/Violence DATA/HockeyFights/Models"
    dataset = args.dataset
    vif_path = args.vif_path
    path_results = args.path_results
    path_violence = args.path_violence
    path_noviolence = args.path_noviolence
    modelType = args.model_type
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    feature_extract = args.feature_extract
    joinType = args.join_type
    scheduler_type = args.scheduler_type
    debugg_mode = args.debugg_mode
    ndis = args.ndis
    path_checkpoints = args.checkpoint_path
    folds_number = args.folds_number
    # path_models = "/media/david/datos/Violence DATA/HockeyFights/Models"
    # path_results = ("/media/david/datos/Violence DATA/HockeyFights/Results/" + dataset_source)
    # path_violence = "/media/david/datos/Violence DATA/HockeyFights/frames/violence"
    # path_noviolence = "/media/david/datos/Violence DATA/HockeyFights/frames/nonviolence"
    # modelType = "alexnet"
    # batch_size = 64
    # num_epochs = 30
    # feature_extract = True
    # joinType = "tempMaxPool"
    # scheduler_type = "OnPlateau"
    # dataset_source = "frames"
    # debugg_mode = False

    # best_acc_test = 0.0
    dataset_source = "frames"
    debugg_mode = False
    avgmaxDuration = 1.66
    interval_duration = 0.3
    num_workers = 4
    input_size = 224

    transforms = createTransforms(input_size)
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    # path_models = '/media/david/datos/Violence DATA/violentflows/Models/'
    # path_results = '/media/david/datos/Violence DATA/violentflows/Results/'+dataset_source
    # gpath = '/media/david/datos/Violence DATA/violentflows/movies Frames'
    init(dataset, vif_path, path_violence, path_noviolence, path_results, path_checkpoints, modelType, ndis, num_workers, transforms,
            dataset_source, interval_duration, avgmaxDuration, batch_size, num_epochs, feature_extract, joinType, scheduler_type,
            device, criterion, folds_number, debugg_mode)

__main__()

