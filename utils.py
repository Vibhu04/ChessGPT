import torch
from torch.utils.data import DataLoader
from dataset import ChessDataset

def load_dataloaders(dataset_path, train_split):

    full_dataset = ChessDataset(dataset_path)
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True,
                            num_workers=0, pin_memory=True)

    return train_loader, val_loader







