
# Install flashtorch

# !pip install flashtorch

# Download example images

# !mkdir -p images


# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import torchvision.models as models

# from flashtorch.utils import apply_transforms, load_image
# from flashtorch.saliency import Backprop

"""## My Model"""

# ! git clone https://github.com/MisaOgura/flashtorch.git

# Load the Drive helper and mount
# from google.colab import drive

# This will prompt for authorization.
# drive.mount('/content/drive')

# !ls "/content/drive/My Drive/VIOLENCE DATASETS"

# !git clone https://github.com/davidGCR/ViolenceModels.git

# !cd ViolenceModels && git pull && pwd


import sys
sys.path.insert(1, 'ViolenceModels-master/')
sys.path.insert(1, 'flashtorch-master/')
from AlexNet import *
from ViolenceDatasetV2 import *
from dinamycImage import *
from util import *
from transforms import *
import torch
from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop


hockey_path_violence = '/media/david/datos/Violence DATA/HockeyFights/frames/violence'
hockey_path_noviolence = '/media/david/datos/Violence DATA/HockeyFights/frames/nonviolence'
datasetAll, labelsAll, numFramesAll = createDataset(hockey_path_violence, hockey_path_noviolence) #ordered
print(len(datasetAll), len(labelsAll), len(numFramesAll))

input_size = 224
data_transforms = createTransforms(input_size)
dataset_source = "frames"
debugg_mode = False
avgmaxDuration = 1.66
interval_duration = 0.3
numDiPerVideos = 1
batch_size = 1
num_workers = 1

model = torch.load('models/alexnet-frames-Finetuned:False-1di-tempMaxPool-OnPlateau.tar')
model.cuda()
backprop = Backprop(model)

image_datasets = {
    "train": ViolenceDatasetVideos( dataset=datasetAll, labels=labelsAll, spatial_transform=data_transforms["train"], source=dataset_source,
        interval_duration=interval_duration,difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, ),
    "test": ViolenceDatasetVideos( dataset=datasetAll, labels=labelsAll, spatial_transform=data_transforms["test"], source=dataset_source,
        interval_duration=interval_duration, difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, )
}
dataloaders_dict = {
    "train": torch.utils.data.DataLoader( image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers, ),
    "test": torch.utils.data.DataLoader( image_datasets["test"], batch_size=batch_size, shuffle=True, num_workers=num_workers, ),
}
for inputs, labels in dataloaders_dict["test"]:
    inputs = inputs.permute(1, 0, 2, 3, 4)
    inputs = inputs.cuda()
    # labels = labels.to(self.device)
    backprop.visualize(inputs, target_class=1, guided=True, use_gpu=True)

"""### 1. Load an image"""

buho = 'images/great_grey_owl.jpg'
di1 = 'images/1.png'
# image = load_image(buho)
image = load_image(buho)

plt.imshow(image)
plt.title('Original image'+str(type(image)))
plt.axis('off');
plt.show()
"""### 2. Load a pre-trained Model"""

# model = models.alexnet(pretrained=True)
# model = torch.load('/content/alexnet-frames-Finetuned:False-1di-tempMaxPool-OnPlateau.tar')

"""### 3. Create an instance of Backprop with the model"""

backprop = Backprop(model)

"""### 4. Visualize saliency maps"""

# Transform the input image to a tensor

owl = apply_transforms(image)
print(owl.size()) #torch.Size([1, 3, 224, 224])
# input_size = 224
# data_transforms = createTransforms(input_size)
# owl = data_transforms['test'](image)
# owl = owl.unsqueeze(dim=0)
# owl = owl.unsqueeze(dim=0)
# owl = owl.permute(1, 0, 2, 3, 4)
# print(owl.size())

# Set a target class from ImageNet task: 24 in case of great gray owl

# target_class = 24

# Ready to roll!

backprop.visualize(owl, target_class=None, guided=True, use_gpu=True)



# """### 5. What about other birds?

# What makes peacock a peacock...?
# """

# peacock = apply_transforms(load_image('/content/images/peacock.jpg'))
# backprop.visualize(peacock, 84, guided=True, use_gpu=True)

# """Or a toucan?"""

# toucan = apply_transforms(load_image('/content/images/toucan.jpg'))
# backprop.visualize(toucan, 96, guided=True, use_gpu=True)

# """Please try out other models/images too!"""