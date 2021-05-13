from train_model import train_model
from data_loader import dataloaders_dict
from vgg16_based import net_vgg16, criterion, optimizer

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    num_epochs = 5
    train_model(net_vgg16, dataloaders_dict, criterion, optimizer, num_epochs)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
