import torch

def verifiParametersToTrain(model):
  params_to_update = model.parameters()
  print("Params to learn:")
  feature_extract = True
  if feature_extract:
      params_to_update = []
      for name,param in model.named_parameters():
          if param.requires_grad == True:
              params_to_update.append(param)
              print("\t",name)
  else:
      for name,param in model.named_parameters():
          if param.requires_grad == True:
              print("\t",name)
  return params_to_update

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']