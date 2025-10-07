#!/bin/bash

# ATTENTION! This automatisation script uses tool telert for automatic notification.
# Read more about this tool you can here: https://github.com/navig-me/telert?tab=readme-ov-file
# If you don't want to use this tool, just rewrite all of the commands without telert using
# For example:
# python3 ./train.py data.num_workers=4 model.backbone=micro_cnn
# python3 ./train.py data.num_workers=4 model.backbone=nano_cnn
# python3 ./train.py data.num_workers=4 model.backbone=tiny_cnn

#==============================================MicroCNN==============================================
telert run --label "train model with backbone MicroCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=MicroCNN
sleep 5
telert run --label "train model with backbone MicroCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=MicroCNN data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=200 data.n_mels=64
sleep 5
telert run --label "train model with backbone MicroCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=MicroCNN data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=64
sleep 5
telert run --label "train model with backbone MicroCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=MicroCNN data.sample_rate=2000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=32
sleep 5
telert run --label "train model with backbone MicroCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=MicroCNN data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=200 data.n_mels=32
sleep 5
telert run --label "train model with backbone MicroCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=MicroCNN data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=128
sleep 5
telert run --label "train model with backbone MicroCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=MicroCNN data.sample_rate=8000 data.n_fft=600 data.hop_length=128 data.win_length=600 data.n_mels=32
sleep 5
#====================================================================================================

#==============================================NanoCNN==============================================
telert run --label "train model with backbone NanoCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=NanoCNN data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=200 data.n_mels=64
sleep 5
telert run --label "train model with backbone NanoCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=NanoCNN data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=64
sleep 5
telert run --label "train model with backbone NanoCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=NanoCNN data.sample_rate=2000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=32
sleep 5
telert run --label "train model with backbone NanoCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=NanoCNN data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=200 data.n_mels=32
sleep 5
telert run --label "train model with backbone NanoCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=NanoCNN data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=128
sleep 5
telert run --label "train model with backbone NanoCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=NanoCNN data.sample_rate=8000 data.n_fft=600 data.hop_length=128 data.win_length=600 data.n_mels=32
sleep 5
telert run --label "train model with backbone NanoCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=NanoCNN
sleep 5
#====================================================================================================

#==============================================TinyCNN==============================================
telert run --label "train model with backbone TinyCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=TinyCNN data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=200 data.n_mels=64
sleep 5
telert run --label "train model with backbone TinyCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=TinyCNN data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=64
sleep 5
telert run --label "train model with backbone TinyCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=TinyCNN data.sample_rate=2000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=32
sleep 5
telert run --label "train model with backbone TinyCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=TinyCNN data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=200 data.n_mels=32
sleep 5
# telert run --label "train model with backbone TinyCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=TinyCNN data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=128
sleep 5
telert run --label "train model with backbone TinyCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=TinyCNN data.sample_rate=8000 data.n_fft=600 data.hop_length=128 data.win_length=600 data.n_mels=32
sleep 5
telert run --label "train model with backbone TinyCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=TinyCNN
sleep 5
#====================================================================================================

#==============================================Conv1dNet==============================================
telert run --label "train model with backbone Conv1dNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=Conv1dNet data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=200 data.n_mels=64
sleep 5
telert run --label "train model with backbone Conv1dNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=Conv1dNet data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=64
sleep 5
telert run --label "train model with backbone Conv1dNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=Conv1dNet data.sample_rate=2000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=32
sleep 5
telert run --label "train model with backbone Conv1dNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=Conv1dNet data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=200 data.n_mels=32
sleep 5
# telert run --label "train model with backbone Conv1dNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=Conv1dNet data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=128
sleep 5
telert run --label "train model with backbone Conv1dNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=Conv1dNet data.sample_rate=8000 data.n_fft=600 data.hop_length=128 data.win_length=600 data.n_mels=32
sleep 5
telert run --label "train model with backbone Conv1dNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=Conv1dNet
sleep 5
#====================================================================================================

#==============================================CustomNet==============================================
telert run --label "train model with backbone CustomNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=CustomNet data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=200 data.n_mels=64
sleep 5
telert run --label "train model with backbone CustomNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=CustomNet data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=64
sleep 5
telert run --label "train model with backbone CustomNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=CustomNet data.sample_rate=2000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=32
sleep 5
telert run --label "train model with backbone CustomNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=CustomNet data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=200 data.n_mels=32
sleep 5
# telert run --label "train model with backbone CustomNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=CustomNet data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=128
sleep 5
telert run --label "train model with backbone CustomNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=CustomNet data.sample_rate=8000 data.n_fft=600 data.hop_length=128 data.win_length=600 data.n_mels=32
sleep 5
telert run --label "train model with backbone CustomNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=CustomNet
sleep 5
#====================================================================================================

#==============================================MobileNetV1_1D==============================================
telert run --label "train model with backbone MobileNetV1_1D $(date)" python3 ./train.py data.num_workers=4 model.backbone=MobileNetV1_1D data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=200 data.n_mels=64
sleep 5
telert run --label "train model with backbone MobileNetV1_1D $(date)" python3 ./train.py data.num_workers=4 model.backbone=MobileNetV1_1D data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=64
sleep 5
telert run --label "train model with backbone MobileNetV1_1D $(date)" python3 ./train.py data.num_workers=4 model.backbone=MobileNetV1_1D data.sample_rate=2000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=32
sleep 5
telert run --label "train model with backbone MobileNetV1_1D $(date)" python3 ./train.py data.num_workers=4 model.backbone=MobileNetV1_1D data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=200 data.n_mels=32
sleep 5
# telert run --label "train model with backbone MobileNetV1_1D $(date)" python3 ./train.py data.num_workers=4 model.backbone=MobileNetV1_1D data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=128
sleep 5
telert run --label "train model with backbone MobileNetV1_1D $(date)" python3 ./train.py data.num_workers=4 model.backbone=MobileNetV1_1D data.sample_rate=8000 data.n_fft=600 data.hop_length=128 data.win_length=600 data.n_mels=32
sleep 5
telert run --label "train model with backbone MobileNetV1_1D $(date)" python3 ./train.py data.num_workers=4 model.backbone=MobileNetV1_1D
sleep 5
#====================================================================================================

#==============================================SimpleMobileNet1D==============================================
telert run --label "train model with backbone SimpleMobileNet1D $(date)" python3 ./train.py data.num_workers=4 model.backbone=SimpleMobileNet1D data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=200 data.n_mels=64
sleep 5
telert run --label "train model with backbone SimpleMobileNet1D $(date)" python3 ./train.py data.num_workers=4 model.backbone=SimpleMobileNet1D data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=64
sleep 5
telert run --label "train model with backbone SimpleMobileNet1D $(date)" python3 ./train.py data.num_workers=4 model.backbone=SimpleMobileNet1D data.sample_rate=2000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=32
sleep 5
telert run --label "train model with backbone SimpleMobileNet1D $(date)" python3 ./train.py data.num_workers=4 model.backbone=SimpleMobileNet1D data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=200 data.n_mels=32
sleep 5
# telert run --label "train model with backbone SimpleMobileNet1D $(date)" python3 ./train.py data.num_workers=4 model.backbone=SimpleMobileNet1D data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=128
sleep 5
telert run --label "train model with backbone SimpleMobileNet1D $(date)" python3 ./train.py data.num_workers=4 model.backbone=SimpleMobileNet1D data.sample_rate=8000 data.n_fft=600 data.hop_length=128 data.win_length=600 data.n_mels=32
sleep 5
telert run --label "train model with backbone SimpleMobileNet1D $(date)" python3 ./train.py data.num_workers=4 model.backbone=SimpleMobileNet1D
sleep 5
#====================================================================================================

#==============================================SmallResNet==============================================
telert run --label "train model with backbone SmallResNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=SmallResNet data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=200 data.n_mels=64
sleep 5
telert run --label "train model with backbone SmallResNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=SmallResNet data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=64
sleep 5
telert run --label "train model with backbone SmallResNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=SmallResNet data.sample_rate=2000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=32
sleep 5
telert run --label "train model with backbone SmallResNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=SmallResNet data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=200 data.n_mels=32
sleep 5
# telert run --label "train model with backbone SmallResNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=SmallResNet data.sample_rate=4000 data.n_fft=400 data.hop_length=128 data.win_length=400 data.n_mels=128
sleep 5
telert run --label "train model with backbone SmallResNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=SmallResNet data.sample_rate=8000 data.n_fft=600 data.hop_length=128 data.win_length=600 data.n_mels=32
sleep 5
telert run --label "train model with backbone SmallResNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=SmallResNet
sleep 5
#====================================================================================================