## Data import
```
kaggle competitions download -c keyword-spotting-mipt-2025
unzip keyword-spotting-mipt-2025.zip
mv keyword-spotting-mipt-2025 data
```

## Train the model
```
python3 train.py
```

Saved best checkpoint you will find in logs/keyword_spotter/version_n/checkpoints

## Check the trian plots via tensorboard
```
tensorboard --logdir ./logs
```

## Convert pretrained model to onnx
```
python3 to_onnx.py ++onnx.checkpoint_path=./logs/keyword_spotter/version_0/checkpoints/best_epoch=08_val_acc=0.90.ckpt
```

## Visualise a model graph via netron
```
netron ./saved_model/model.onnx
```
