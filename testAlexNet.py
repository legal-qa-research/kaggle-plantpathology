from torchvision import models
import torch
from data_loader import dataloaders_dict
import numpy as np
import pandas as pd


resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

net_alexnet = models.alexnet(pretrained=True)

f = open('./submission.txt', 'w')
df_pred_list = []
for inputs, _, image_name in dataloaders_dict['test']:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_alexnet.to(device)
    inputs = inputs.to(device)
    out = net_alexnet(inputs)
    out = out.to(device)
    maxid = np.argmax(out.detach().numpy(), axis=1)
    df_pred = pd.DataFrame({'labels': pd.Series(maxid), "image": pd.Series(image_name)})
    df_pred_list.append(df_pred)

df_submission = pd.concat(df_pred_list, axis=0)
df_submission = df_submission[["image", "labels"]].reset_index(drop=True)
df_submission.to_csv('./submission.csv', index=False)

print(df_pred_list)

f.close()

