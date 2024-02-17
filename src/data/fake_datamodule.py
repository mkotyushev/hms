import torch
from lightning.pytorch.demos.boring_classes import BoringDataModule


class FakeDataModule(BoringDataModule):
    def train_dataloader(self):
        print("⚡", "using FakeDataset", "⚡")
        return torch.utils.data.DataLoader(self.random_train)
