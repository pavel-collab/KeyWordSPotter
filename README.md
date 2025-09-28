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
