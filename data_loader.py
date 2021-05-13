import torch.utils.data as data
import pandas as pd
from data_preprocess import to_label, make_datapath_list
from constant import mean, std, size, TRAIN_CSV_PATH, SAMPLE_SUBMISSION_PATH

# Load Header
from image_transform import ImageTransform
from plant_dataset import PlantDataset

df_train = pd.read_csv(TRAIN_CSV_PATH)
df_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)

d_set = set()
for k in df_train.labels.unique():
    d_set = d_set | set(k.split(" "))
print(f"num of labels: {len(d_set)}  {d_set}")

df_train = to_label(df_train)
# df_labels_idx = df_train.loc[df_train.duplicated(["labels", "labels_n"]) is False][["labels_n", "labels"]].set_index("labels_n").sort_index()
df_labels_idx = df_train.loc[df_train.duplicated(["labels", "labels_n"])==False]\
                [["labels_n", "labels"]].set_index("labels_n").sort_index()

# Load Image
train_list = make_datapath_list(phase="train")
print(f"train data length : {len(train_list)}")
val_list = make_datapath_list(phase="val")
print(f"validation data length : {len(val_list)}")
test_list = make_datapath_list(phase="test")
print(f"test data length : {len(test_list)}")

# Create Dataset
train_dataset = PlantDataset(df_labels_idx, df_train, train_list, transform=ImageTransform(size, mean, std)
                             , phase='train')
val_dataset = PlantDataset(df_labels_idx, df_train, val_list, transform=ImageTransform(size, mean, std), phase='val')
test_dataset = PlantDataset(df_labels_idx, df_train, test_list, transform=ImageTransform(size, mean, std), phase='test')

index = 0

print("【train dataset】")
print(f"img num : {train_dataset.__len__()}")
print(f"img : {train_dataset.__getitem__(index)[0].size()}")
print(f"label : {train_dataset.__getitem__(index)[1]}")
print(f"image name : {train_dataset.__getitem__(index)[2]}")

print("\n【validation dataset】")
print(f"img num : {val_dataset.__len__()}")
print(f"img : {val_dataset.__getitem__(index)[0].size()}")
print(f"label : {val_dataset.__getitem__(index)[1]}")
print(f"image name : {val_dataset.__getitem__(index)[2]}")

print("\n【test dataset】")
print(f"img num : {test_dataset.__len__()}")
print(f"img : {test_dataset.__getitem__(index)[0].size()}")
print(f"label : {test_dataset.__getitem__(index)[1]}")
print(f"image name : {test_dataset.__getitem__(index)[2]}")

batch_size = 128

# Create DataLoader
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# to Dictionary
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}
