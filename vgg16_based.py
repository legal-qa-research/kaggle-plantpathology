# from data_loader import
from torchvision import models
import torch.nn as nn
import torch.optim as optim

# Load the learned VGG-16 model.

# Create an instance of the VGG-16 model
use_pretrained = True
net_vgg16 = models.vgg16(pretrained=use_pretrained)
#
# load_path = "/kaggle/input/pytorch-pretrained-models/vgg16-397923af.pth"
# if torch.cuda.is_available():
#     load_weights = torch.load(load_path)
#     net_vgg16.load_state_dict(load_weights)
# else:
#     load_weights = torch.load(load_path, map_location={"cuda:0": "cpu"})
#     net_vgg16.load_state_dict(load_weights)

# Replace the output unit of the last output layer of the VGG-16 model.
# out_features 1000 to 12
net_vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=12)

# Set to training mode.
net_vgg16.train()

criterion = nn.CrossEntropyLoss()

# Store the parameters to be learned by finetuning in the variable params_to_update.
params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

# Specify the parameter name of the layer to be trained.
update_param_names_1 = ["features.24.weight", "features.24.bias", "features.26.weight", "features.26.bias",
                        "features.28.weight", "features.28.bias"]
update_param_names_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

for name, param in net_vgg16.named_parameters():
    if name in update_param_names_1:
        param.requires_grad = False
        params_to_update_1.append(param)
        print(f"Store in params_to_update_1 : {name}")
    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        print(f"Store in params_to_update_2 : {name}")
    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
        print(f"Store in params_to_update_3 : {name}")
    else:
        param.requires_grad = False
        print(f"Parameters not to be learned :  {name}")

# Set Optimizer
optimizer = optim.SGD([
    {"params": params_to_update_1, "lr": 1e-4},
    {"params": params_to_update_2, "lr": 5e-4},
    {"params": params_to_update_3, "lr": 1e-3}
], momentum=0.9)
