from torchvision import models
import torch.nn as nn
import torch.optim as optim


net_alexnet = models.alexnet(pretrained=False)
#
# if torch.cuda.is_available():
#     load_weights = torch.load(load_path_alexNet)
#     net_alexnet.load_state_dict(load_weights)
# else:
#     load_weights = torch.load(load_path_alexNet, map_location={"cuda:0": "cpu"})
#     net_alexnet.load_state_dict(load_weights)

# Replace the output unit of the last output layer of the VGG-16 model.
# out_features 1000 to 12
net_alexnet.classifier[6] = nn.Linear(in_features=4096, out_features=12)

# Set to training mode.
net_alexnet.train()

criterion = nn.CrossEntropyLoss()

# Store the parameters to be learned by finetuning in the variable params_to_update.
params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

# Specify the parameter name of the layer to be trained.
update_param_names_1 = ["features.0.weight", "features.0.bias","features.3.weight", "features.3.bias", "features.6.weight", "features.6.bias", "features.8.weight", "features.8.bias", "features.10.weight", "features.10.bias"]
update_param_names_2 = ["classifier.1.weight", "classifier.1.bias", "classifier.4.weight", "classifier.4.bias"]
update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]


for name, param in net_alexnet.named_parameters():
    if name in update_param_names_1:
        param.requires_grad = True
        params_to_update_1.append(param)
        if param.requires_grad:
            print(f"Store in params_to_update_1 : {name}")
        else:
            print(f"Parameters not to be learned :  {name}")
    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        if param.requires_grad:
            print(f"Store in params_to_update_2 : {name}")
        else:
            print(f"Parameters not to be learned :  {name}")
    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
        if param.requires_grad:
            print(f"Store in params_to_update_3 : {name}")
        else:
            print(f"Parameters not to be learned :  {name}")
    else:
        param.requires_grad = False
        print(f"Parameters not to be learned :  {name}")

# Set Optimizer
optimizer = optim.SGD([
    {"params": params_to_update_1, "lr": 1e-3},
    {"params": params_to_update_2, "lr": 1e-3},
    {"params": params_to_update_3, "lr": 1e-3}
], momentum=0.9)
