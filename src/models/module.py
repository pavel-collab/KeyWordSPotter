import pytorch_lightning as pl
import torch.nn.functional as F
from torch import optim
from torchmetrics import Accuracy
import torch 
from typing import Tuple
import torch.nn as nn

from src.models.utils import model_map

class KeywordSpotter(pl.LightningModule):
    def __init__(self, num_classes=2, backbone="ResNet18Backbone", in_features=64, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
                
        if backbone not in model_map.keys():
            raise RuntimeError(f"There are no known backbone model {backbone}")
        self.model = model_map[backbone](num_classes, in_features=in_features)
        
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
    
from src.models.microcnn import MicroCNN
# from src.models.backbone import ResNet18Backbone

class KDLitModule(pl.LightningModule):
    def __init__(self, num_classes: int = 2, teacher_checkpoint_path: str = None, in_features: int = 1,
                 lr: float = 1e-3, weight_decay: float = 1e-2,
                 alpha: float = 0.65, T: float = 3.0, label_smoothing: float = 0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["student","teacher"])

        #! temporary hardcode student arch 
        self.student = MicroCNN(in_features=64, num_classes=num_classes)
        
        TModule = KeywordSpotter(num_classes=num_classes, backbone="ResNet18Backbone")
        #! temporary hardcode teacher arch
        if teacher_checkpoint_path is not None:
            ckpt = torch.load(teacher_checkpoint_path, map_location="cpu", weights_only = False)
            TModule.load_state_dict(
                {k: v for k, v in ckpt["state_dict"].items() if "total" not in k}
            )
        # using it without checkpoitn is relevant if we're in evaluation mode, when we need to use only student model

        self.teacher = TModule.model.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.val_loss = torch.nn.CrossEntropyLoss()

        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, x):
        return self.student(x)

    def kd_loss(self, s_logits, t_logits):
        T = self.hparams.T
        log_p_s = torch.log_softmax(s_logits / T, dim=1)
        p_t     = torch.softmax(t_logits / T, dim=1)
        return self.kl(log_p_s, p_t) * (T * T)

    def training_step(self, batch, _):
        # _, inputs, labels = batch 
        x, y = batch               
        # s_logits, preds = self.forward(inputs)  
        s_logits = self.forward(x) 
        with torch.no_grad():
            t_logits = self.teacher(x)

        ce = self.ce(s_logits, y)
        kd = self.kd_loss(s_logits, t_logits)
        alpha = self.hparams.alpha
        loss = alpha * ce + (1.0 - alpha) * kd

        log = {
            "train/loss": loss,
            "lr": self.optimizers().param_groups[0]["lr"],
            "train/accuracy": self.train_accuracy(s_logits, y),
        }

        self.log_dict(log, on_step=True)

        return {"loss": loss}

    # def on_train_epoch_end(self):
    #     self.train_acc.reset()

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        # _, inputs, labels = batch
        x, y = batch 
        logits = self.forward(x)

        # logprobs, preds = self.forward(inputs)

        loss = F.cross_entropy(logits, y)
        # self.valid_acc.update(preds, labels)
        val_acc = self.val_accuracy(logits, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)

        return {"val_loss": loss, "val_accuracy": val_acc}

    def configure_optimizers(self):
        wd = 0.01
        lr = 1e-3
        betas = (0.9, 0.98)

        wd_params, no_wd_params = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad: 
                continue
            if p.ndim == 1 or n.endswith(".bias") or "bn" in n.lower() or "batchnorm" in n.lower():
                no_wd_params.append(p)
            else:
                wd_params.append(p)

        opt = torch.optim.AdamW(
            [{"params": wd_params, "weight_decay": wd},
             {"params": no_wd_params, "weight_decay": 0.0}],
            lr=lr, betas=betas
        )

        sched = torch.optim.lr_scheduler.OneCycleLR(max_lr=2.5e-3,
          total_steps= 32000,
          pct_start= 0.1,
          anneal_strategy= "cos",
          div_factor= 10,  
          final_div_factor= 100, 
          optimizer=opt)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}