from train_model import train_model
from data_loader import dataloaders_dict
# from vgg16_based import net_vgg16, criterion, optimizer
from alexnet_based import net_alexnet, criterion, optimizer
import torch
from load_weight import load_weight
from constant import PRETRAIN_MODEL_PATH, SAVE_MODEL_WEIGHT_PATH

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    num_epochs = 10
    # train_model(net_vgg16, dataloaders_dict, criterion, optimizer, num_epochs)
    load_weight(net_alexnet, PRETRAIN_MODEL_PATH)
    train_model(net_alexnet, dataloaders_dict, criterion, optimizer, num_epochs)
    save_path = SAVE_MODEL_WEIGHT_PATH
    torch.save(net_alexnet.state_dict(), save_path)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
