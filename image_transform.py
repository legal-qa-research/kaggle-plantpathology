from torchvision import models, transforms


class ImageTransform:
    """
    Class for image preprocessing.

    Attributes
    ----------
    resize : int
        224
    mean : (R, G, B)
        Average value for each color channel
    std : (R, G, B)
        Standard deviation for each color channel
    """

    def __init__(self, resize, mean, std):
        self.data_transform = {
            #             'train': A.Compose(albumentation_list),
            'train': transforms.Compose([
                transforms.Resize(resize),
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomPerspective(),
                transforms.ToTensor(),
                #                 transforms.RandomRotation(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase="train"):
        """
        Parameters
        ----------
        phase: 'train' or 'val' or 'test'
            Specify the mode of preprocessing
        """

        return self.data_transform[phase](img)
#         return self.data_transform[phase](image=img).get('image')