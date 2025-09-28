import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim
from torchmetrics import Accuracy
from src.models.backbone import CNNBackbone

class KeywordSpotter(pl.LightningModule):
    def __init__(self, num_classes=2, backbone="resnet18", learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
                
        self.model = CNNBackbone(backbone, num_classes)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.accuracy(logits, y), prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch        
        logits = self(x)
                
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }