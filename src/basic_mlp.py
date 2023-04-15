from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets
import torch.nn as nn
import torch
torch.set_float32_matmul_precision('medium')

from argparse import ArgumentParser

import lightning.pytorch as pl
pl.seed_everything(1701, workers=True)

import matplotlib.pyplot as plt


class NeuralNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = 1e-3
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = self.loss_fn(z, y)
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = self.loss_fn(z, y)
        self.log("valid_loss", loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":

    # Read in arguments
    parser = ArgumentParser()
    args = parser.parse_args()
    
    DATA_PARAMS = dict(root = "rsc", download=True, transform=ToTensor())
    DATALOADER_PARAMS = dict(batch_size = 64, num_workers=16)

    training_data = datasets.MNIST(train=True, shuffle=True, **DATA_PARAMS)
    test_data = datasets.MNIST(train=False, **DATA_PARAMS)
    print(test_data)
    
    training_dl = DataLoader(training_data, **DATALOADER_PARAMS)
    test_dl = DataLoader(test_data, **DATALOADER_PARAMS)

    model = NeuralNetwork()
    print(model)

    trainer = pl.Trainer(max_epochs=5, accelerator="auto")
    trainer.fit(model=model, train_dataloaders=training_dl, val_dataloaders=test_dl)
    print(trainer)
    