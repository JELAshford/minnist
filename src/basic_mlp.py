from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch
torch.set_float32_matmul_precision('medium')

import lightning.pytorch as pl
pl.seed_everything(1701, workers=True)

import matplotlib.pyplot as plt


class NeuralNetwork(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3):
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
        self.learning_rate = learning_rate
        self.save_hyperparameters()
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
    def predict_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        return z
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":

    CHECKPOINT_PATH = None
    
    DATA_PARAMS = dict(root = "rsc", download=True, transform=transforms.ToTensor())
    DATALOADER_PARAMS = dict(batch_size = 64, num_workers=16)

    # Load dataset
    training_data = datasets.MNIST(train=True, **DATA_PARAMS)
    test_data = datasets.MNIST(train=False, **DATA_PARAMS)
    
    training_dl = DataLoader(training_data, shuffle=True, **DATALOADER_PARAMS)
    test_dl = DataLoader(test_data, **DATALOADER_PARAMS)

    # Load or train model
    if CHECKPOINT_PATH is not None:
        model = NeuralNetwork.load_from_checkpoint(CHECKPOINT_PATH)
    else: 
        model = NeuralNetwork()
        trainer = pl.Trainer(max_epochs=5, accelerator="auto", default_root_dir="out/")
        trainer.fit(model=model, train_dataloaders=training_dl)
    print(model)
    
    # Predictions
    out = torch.cat(trainer.predict(model, test_dl))
    preds = torch.argmax(out, axis=1)
    labels = test_dl.dataset.targets
    print(f"accuracy = {sum(preds == labels) / len(labels):.3f}")