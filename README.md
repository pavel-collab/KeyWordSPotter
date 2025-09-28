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
ATTENTION 1: notice, that runa = is a special runa for this command. So, we have to masking it by charackter \ in the line.
It can be a little bit uncomfortable. Maybe, we can save checkpoint in a different format.
ATTENTION 2: please, don't forget to set a correct backbone model name.
```
python3 to_onnx.py 'onnx.checkpoint_path=./logs/keyword_spotter/version_0/checkpoints/best_epoch\=08_val_acc\=0.90.ckpt' model.backbone=baseline
```

## Visualise a model graph via netron
```
netron ./saved_model/model.onnx
```

## Make a prediction on the saved onnx model
Notice: all of the dataloaders have been already set to import test data for prediction.
ATTENTION: please, don't forget to set a correct saved model name.
```
python3 ./predict.py onnx.onnx_file_name=model_baseline_2025-09-28_22-53-54.onnx
```

## Submit prediction to kaggle
```
kaggle competitions submit -c keyword-spotting-mipt-2025 -f submission.csv -m "Message"
```