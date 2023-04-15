from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch
torch.set_float32_matmul_precision('medium')

import matplotlib.pyplot as plt

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl
pl.seed_everything(1701, workers=True)
import wandb 



class NeuralNetwork(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3, l1_loss_weight = 1e-2):
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
        self.l1_loss_weight = l1_loss_weight
        self.save_hyperparameters()
    def linear_l1_loss(self):
        total_val = 0.
        for layer in self.linear_relu_stack:
            if isinstance(layer, nn.modules.linear.Linear):
                # Minimise weights
                total_val += torch.mean(torch.sum(layer.weight.abs(), axis=1))
                # Minimise biases
                total_val += torch.sum(layer.bias.abs())
        return total_val
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = self.loss_fn(z, y) + self.l1_loss_weight * self.linear_l1_loss()
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
    
    DATA_PARAMS = dict(root = "rsc", download=True, transform=transforms.ToTensor())
    DATALOADER_PARAMS = dict(batch_size = 64, num_workers = 16)

    # Load dataset
    training_data = datasets.MNIST(train=True, **DATA_PARAMS)
    test_data = datasets.MNIST(train=False, **DATA_PARAMS)
    
    training_dl = DataLoader(training_data, shuffle=True, **DATALOADER_PARAMS)
    test_dl = DataLoader(test_data, **DATALOADER_PARAMS)

    # Setup loggin/training
    wandb_logger = WandbLogger(name="l1_testing", project="minnist", group="mlp", save_dir = "out/", dir="out/", log_model = True)
    trainer = pl.Trainer(max_epochs=50, accelerator="auto", default_root_dir="out/", 
        logger=wandb_logger, callbacks=[EarlyStopping(monitor="valid_loss", mode="min", patience=5)])
    
    # Train model
    model = NeuralNetwork(l1_loss_weight=5e-2)
    trainer.fit(model=model, train_dataloaders=training_dl, val_dataloaders=test_dl)

    # Print weight magnitudes
    plt.hist(model.linear_relu_stack[0].weight.detach().numpy().flatten(), bins=20)    
    plt.show()
    plt.imshow(model.linear_relu_stack[0].weight.detach().numpy())
    plt.show()
    
    # Predictions
    out = torch.cat(trainer.predict(model, test_dl))
    preds = torch.argmax(out, axis=1)
    labels = test_dl.dataset.targets
    print(f"accuracy = {sum(preds == labels) / len(labels):.3f}")