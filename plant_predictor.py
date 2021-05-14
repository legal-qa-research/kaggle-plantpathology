import pandas as pd
import numpy as np
import torch
from tqdm.notebook import tqdm


class PlantPredictor:
    """
    Class for predicting labels from output results

    Attributes
    ----------
    df_labels_idx: DataFrame
        DataFrame that associates INDEX with a label name
    """

    def __init__(self, net, df_labels_idx, dataloaders_dict):
        self.net = net
        self.df_labels_idx = df_labels_idx
        self.dataloaders_dict = dataloaders_dict
        self.df_submit = pd.DataFrame()

    def __predict_max(self, out):
        """
        Get the label name with the highest probability.

        Parameters
        ----------
        predicted_label_name: str
            Name of the label with the highest prediction probability
        """
        maxid = np.argmax(out.detach().numpy(), axis=1)
        df_predicted_label_name = self.df_labels_idx.iloc[maxid]

        return df_predicted_label_name

    def inference(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Devices to be used : {device}")
        df_pred_list = []
        for inputs, _, image_name in tqdm(self.dataloaders_dict['test']):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.net.to(device)
            inputs = inputs.to(device)
            out = self.net(inputs)
            device = torch.device("cpu")
            out = out.to(device)
            df_pred = self.__predict_max(out).reset_index(drop=True)
            df_pred["image"] = image_name
            df_pred_list.append(df_pred)

        self.df_submit = pd.concat(df_pred_list, axis=0)
        self.df_submit = self.df_submit[["image", "labels"]].reset_index(drop=True)

