import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from src.models.module import KeywordSpotter
from src.dataset.module import AudioDataModule
from pytorch_lightning import seed_everything

@hydra.main(version_base=None, config_path="./configs", config_name="kws.yaml")
def main(cfg: DictConfig):
    seed_everything(314, workers=True)
    
    # Инициализация
    model = KeywordSpotter(
        num_classes=cfg.model.num_classes,
        backbone=cfg.model.backbone,
        learning_rate=cfg.model.learning_rate
    )
    
    data_module = AudioDataModule(cfg.data)
    logger = TensorBoardLogger("logs", name="keyword_spotter")

    # Тренер
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                save_top_k=1,
                filename="best_{epoch:02d}_{val_acc:.2f}"
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_acc",
                patience=5,
                mode="max"
            )
        ]
    )

    # Обучение
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()