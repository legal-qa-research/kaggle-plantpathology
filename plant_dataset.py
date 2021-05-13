import torch.utils.data as data
from PIL import Image


class PlantDataset(data.Dataset):
    """
    Class to create a Dataset

    Attributes
    ----------
    df_train : DataFrame
        DataFrame containing the image labels.
    file_list : list
        A list containing the paths to the images
    transform : object
        Instance of the preprocessing class (ImageTransform)
    phase : 'train' or 'val' or 'test'
        Specify whether to use train, validation, or test
    """

    def __init__(self, df_labels_idx, df_train, file_list, transform=None, phase='train'):
        self.df_train = df_train
        self.df_labels_idx = df_labels_idx
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        """
        Returns the number of images.
        """
        return len(self.file_list)

    def __getitem__(self, index):
        """
        Get data in Tensor format and labels of preprocessed images.
        """
        # print(index)

        # Load the index number image.
        img_path = self.file_list[index]
        img = Image.open(img_path)
        #         img = cv2.imread(img_path)
        #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocessing images
        img_transformed = self.transform(img, self.phase)

        # image name
        image_name = img_path[-20:]

        # Extract the labels
        if self.phase in ["train", "val"]:
            label = self.df_train.loc[self.df_train["image"] == image_name]["labels_n"].values[0]
        elif self.phase in ["test"]:
            label = -1

        return img_transformed, label, image_name
