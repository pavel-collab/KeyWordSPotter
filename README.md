## Data import
```
kaggle competitions download -c keyword-spotting-mipt-2025
unzip keyword-spotting-mipt-2025.zip
mv keyword-spotting-mipt-2025 data
```

## Data augmantation
```commandline
python3 augmentation.py -m ./data/train/manifest_train.csv
```
This step is not nesessary. If you use augmentation, don't
forget to set a new path to a train manifest.

## Train the model
```
python3 train.py
```
Saved best checkpoint you will find in logs/keyword_spotter/version_n/checkpoints

### Distilation
There is a special option for distilate small model from large one. For example, we can distilate MicroCNN from 
ResNet18. To do it, you will do two steps. Firstly you learn large model, for example ResNet18.
```
python3 train.py model.backbone=ResNet18Backbone pipeline.check_limits=false
```
Don't forget to disable check limits option, overwise your script will fall with error, because large models are out 
of competition limits (parameters and MACs). After training you need to save checkpoint of your model.

After that you can learn student model, based of large model, you have trained.
```
python3 train.py model.student=true onnx.checkpoint_path='./logs/keyword_spotter/version_2/checkpoints/best_epoch\=07_val_acc\=0.91.ckpt'
```
Set a path to your large model checkpoint. And don't forget set an option model.student.

## Check the trian plots via tensorboard
```
tensorboard --logdir ./logs
```

## Convert pretrained model to onnx
ATTENTION: notice, that runa = is a special runa for this command. So, we have to masking it by charackter \ in the line.
It can be a little bit uncomfortable. Maybe, we can save checkpoint in a different format.
```
python3 to_onnx.py onnx.checkpoint_path=./logs/keyword_spotter/version_0/checkpoints/best_epoch\=08_val_acc\=0.90.ckpt model.backbone=SimpleMobileNet1D
```

## Visualise a model graph via netron
```
netron ./saved_model/model.onnx
```

## Validate model
You can run your saved onnx model on validation dataset and check out the classification report 
local. To do it run
```
python3 ./validate.py onnx.onnx_file_name=<path to onnx model>
```

## Make a prediction on the saved onnx model
Notice: all of the dataloaders have been already set to import test data for prediction.
```
python3 ./predict.py onnx.onnx_file_name=<path to onnx model>
```

## Submit prediction to kaggle
```
kaggle competitions submit -c keyword-spotting-mipt-2025 -f submission.csv -m "Message"
```