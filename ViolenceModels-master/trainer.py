import torch
import torchvision
# from tensorboardcolab import TensorBoardColab
import time
import copy
from util import save_checkpoint


class Trainer:
    def __init__(self, model, dataloaders, criterion, optimizer, scheduler, device, num_epochs, checkpoint_path):
        self.model = model
        # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
        # self.model_name = "alexnet"
        # Number of classes in the dataset
        self.num_classes = 2
        
        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        self.feature_extract = True

        self.input_size = 224
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.criterion = criterion
        # self.tb = TensorBoardColab()
        self.device = device
        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_acc = 0.0

    def train_epoch(self, epoch):
        # self.scheduler.step(epoch)
        self.model.train()  # Set model to training mode
        # is_inception = False
        running_loss = 0.0
        running_corrects = 0
        # Iterate over data.
        for inputs, labels in self.dataloaders["train"]:
            # print('==== dataloader size: ',inputs.size()) #[batch, ndi, ch, h, w]
            inputs = inputs.permute(1, 0, 2, 3, 4)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # track history if only in train
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                # Special case for inception because in training it has an auxiliary output. In train
                #   mode we calculate the loss by summing the final output and the auxiliary output
                #   but in testing we only consider the final output.
                # if is_inception:
                #     outputs, aux_outputs = self.model(inputs)
                #     loss1 = self.criterion(outputs, labels)
                #     loss2 = self.criterion(aux_outputs, labels)
                #     loss = loss1 + 0.4 * loss2
                # else:
                    
                outputs = self.model(inputs)
                # print('-- outputs size: ', outputs.size())
                # print('-- labels size: ',labels.size())
                loss = self.criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                # backward + optimize only if in training phase

                loss.backward()
                self.optimizer.step()
                # self.scheduler.step(epoch)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(self.dataloaders["train"].dataset)
        epoch_acc = running_corrects.double() / len(self.dataloaders["train"].dataset)

        print("{} Loss: {:.4f} Acc: {:.4f}".format('train', epoch_loss, epoch_acc))
        # self.tb.save_value("trainLoss", "train_loss", epoch, epoch_loss)
        # self.tb.save_value("trainAcc", "train_acc", epoch, epoch_acc)
        return epoch_loss, epoch_acc

    def test_epoch(self, epoch):
        running_loss = 0.0
        running_corrects = 0
        self.model.eval()
        
        # Iterate over data.
        for inputs, labels in self.dataloaders["test"]:
            inputs = inputs.permute(1, 0, 2, 3, 4)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                # print('-- outputs size: ', outputs.size())
                # print('-- labels size: ',labels.size())
                loss = self.criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(self.dataloaders["test"].dataset)
        epoch_acc = running_corrects.double() / len(self.dataloaders["test"].dataset)

        print("{} Loss: {:.4f} Acc: {:.4f}".format("test", epoch_loss, epoch_acc))
        if self.checkpoint_path != None and epoch_acc > self.best_acc:
            self.best_acc = epoch_acc
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            save_checkpoint(self.model, self.checkpoint_path)
        # self.tb.save_value("testLoss", "test_loss", epoch, epoch_loss)
        # self.tb.save_value("testAcc", "test_acc", epoch, epoch_acc)

        return epoch_loss, epoch_acc
