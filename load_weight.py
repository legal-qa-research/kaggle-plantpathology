import torch


def load_weight(net, load_path):
    # load_path = "/kaggle/input/d/kuboko/plantpathology2021/vgg16_fine_tuning_v1.h"
    if torch.cuda.is_available():
        load_weights = torch.load(load_path)
        net.load_state_dict(load_weights)
    else:
        load_weights = torch.load(load_path, map_location={"cuda:0": "cpu"})
        net.load_state_dict(load_weights)
