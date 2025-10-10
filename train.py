import hydra
from omegaconf import DictConfig
import thop
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from src.models.module import KeywordSpotter, KDLitModule
from src.dataset.module import AudioDataModule
from pytorch_lightning import seed_everything
import torch
from utils import check_params

import warnings

from logger import local_logger

# turn off the UserWarnings because lots of them are talking about 
# library function refactoring, last or future deprecations
warnings.filterwarnings('ignore', category=UserWarning)

@hydra.main(version_base=None, config_path="./configs", config_name="kws.yaml")
def main(cfg: DictConfig):
    seed_everything(314, workers=True)
    
    if cfg.pipeline.check_params:
        errors = check_params(
            sample_rate=cfg.data.sample_rate,
            n_fft=cfg.data.n_fft,
            win_length=cfg.data.win_length,
            hop_length=cfg.data.hop_length,
            n_mels=cfg.data.n_mels
        )

        if len(errors) != 0:
            header_str = f"Params: sample_rate={cfg.data.sample_rate}, " \
                        f"n_fft={cfg.data.n_fft}, " \
                        f"win_length={cfg.data.win_length}, " \
                        f"hop_length={cfg.data.hop_length}, " \
                        f"n_mels={cfg.data.n_mels}" \
                        " "
            for error in errors:
                local_logger.critical(header_str + error)
                return

    # Инициализация
    if not cfg.model.student:
        model = KeywordSpotter(
            num_classes=cfg.model.num_classes,
            backbone=cfg.model.backbone,
            learning_rate=cfg.model.learning_rate,
            in_features=cfg.model.in_features,
        )
    elif cfg.distilation.teacher_checkpoint_path is not None:
        model = KDLitModule(
            num_classes=cfg.model.num_classes,
            teacher_checkpoint_path=cfg.distilation.teacher_checkpoint_path,
            lr=cfg.model.learning_rate,
            in_features=cfg.model.in_features,
        )
    else:
        local_logger.critical("you're trying to train student model, but checkpoint_path is not specified")
        return
    
    data_module = AudioDataModule(cfg.data)
    logger = TensorBoardLogger("logs", name="keyword_spotter")
    
    # Check an amount of trainable parameters and MACs
    data_module = AudioDataModule(cfg.data)
    data_module.setup()
    dataloader = data_module.train_dataloader()
    
    for batch in dataloader:
        # we're getting batch from train dataloader, that returns a batch of data and labels
        # the first element of batch (batch[0]) is a batch of data (3 dimention for audio)
        # the second one is a batch of labels (batch[1]) -- one dimention tensor
        # So, here to obtain a shape of input data, we have to take the batch of data batch[0]
        # and next take the first element in this batch -- batch[0][0]
        # input_dim = batch[0][0].shape
        
        ## We check the number of MAC only on one element (batch_size = 1),
        ## but train model on the batch_size != 1
        ## in the input_dim we finally have to have 3 dimention
        if batch[0].size(0) != 1:
            input_dim = batch[0][0][None, :, :].shape
        else:
            input_dim = batch[0].shape            
        break
    
    # Создание примера входных данных
    sample_inputs = torch.randn(input_dim)
    
    if not cfg.model.student:
        torch_model = model.model
    else:
        torch_model = model.student

    macs, params = thop.profile(
        # here we have to use a backbone model with classification head, not a pytorch loghtning wrapper
        torch_model, 
        inputs=(sample_inputs,),
    )
    
    local_logger.info(f"MACs: {macs}")
    local_logger.info(f"Params: {params}")
    
    if cfg.pipeline.check_limits:
        if macs > 1e6:
            local_logger.critical(f"The number of multiply-accumulate operations for model {cfg.model.backbone} is grater than available limit {macs}>1e6")
            return
        if params > 1e4:
            local_logger.critical(f"The number of parameters for model {cfg.model.backbone} is grater than available limit {macs}>1e4")
            return

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