from torchvision import models
import torch
from data_loader import dataloaders_dict
from tqdm.notebook import tqdm
import numpy as np


resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

net_alexnet = models.alexnet(pretrained=True)

f = open('./submission.txt', 'w')
for inputs, _, image_name in tqdm(dataloaders_dict['test']):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_alexnet.to(device)
    inputs = inputs.to(device)
    out = net_alexnet(inputs)
    out = out.to(device)
    maxid = np.argmax(out.detach().numpy(), axis=1)
    for id_label in maxid:
        print (id_label)
        f.write(f'{id_label}\n')

f.close()

