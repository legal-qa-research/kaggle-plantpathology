import torch
from tqdm import tqdm


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    """
    Function for training the model.

    Parameters
    ----------
    net: object
    dataloaders_dict: dictionary
    criterion: object
    optimizer: object
    num_epochs: int
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Devices to be used : {device}")
    net.to(device)
    torch.backends.cudnn.benchmark = True
    # loop for epoch
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} / {num_epochs}")
        print("-------------------------------")
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()
            epoch_loss = 0.0
            epoch_corrects = 0
            # if (epoch == 0) and (phase == "train"):
            # continue

            pbar = tqdm(total=len(dataloaders_dict[phase]))

            for inputs, labels, _ in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
                pbar.update(1)
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")