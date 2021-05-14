import os.path as osp
from constant import TRAIN_IMAGE_PATH, TEST_IMAGE_PATH
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def make_datapath_list(phase="train", val_size=0.25):
    """
    Function to create a PATH to the data.

    Parameters
    ----------
    phase : 'train' or 'val' or 'test'
        Specify whether to use Train data or test data.
    val_size : float
        Ratio of validation data to train data

    Returns
    -------
    path_lsit : list
        A list containing the PATH to the data.
    """

    if phase in ["train", "val"]:
        phase_path = "train_images"
    elif phase in ["test"]:
        phase_path = "test_images"
    else:
        print(f"{phase} not in path")
    target_path = osp.join(TRAIN_IMAGE_PATH, '*.jpg') if phase in ['train', 'val'] \
        else osp.join(TEST_IMAGE_PATH, '*.{jpg,JPEG}')

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    if phase in ["train", "val"]:
        train, val = train_test_split(path_list, test_size=val_size, random_state=0, shuffle=True)
        if phase == "train":
            path_list = train
        else:
            path_list = val

    return path_list


def to_label(df):
    """
    Function for Label encoding.
    """
    le = LabelEncoder()
    df["labels_n"] = le.fit_transform(df.labels.values)
    return df
