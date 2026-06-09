# 实验记录

## 第一次

全部启动，pca 似乎破坏了 emotion 到 clip 语义空间的映射，导致生成图像中，没有出现限制与情感相关的对象与场景

第一版，可能 pca 导致 attr_loss 失效，语义映射失败
                    loss_forward = (
                         (1 - attr_rate) * (loss_reconstruction)
                         + args.attr_rate * attr_rate * loss_attr
                         + args.emo_rate * loss_emo
                         + loss_fe
                     )

```
attr_rate: 0.01
emo_rate: 0
emotion: all
learnable_property: ["object","scene"]
learning_rate: 0.001
# max_train_steps: 100
max_train_steps: 100000
model: MLP
need_Dropout: false
need_LN: false
need_ReLU: true
num_fc_layers: 2
num_train_epochs: 10
output_dir: runs/my_test
pretrained_model_name_or_path: /mnt/d/model/stable-diffusion-v1-5/

# pretrained_model_name_or_path: stable-diffusion-v1-5/stable-diffusion-v1-5
seed: 520
threshold: 0
train_data_dir: /mnt/d/dataset/EmoSet/LableSplit_20260529_074548/
# train_data_dir: /mnt/d/dataset/EmoSet/0103_split_to_folder/

# Append these keys to the original config/config.yaml.
# Start conservative; increase weights only after verifying loss curves and samples.

fe_use: true

# Feature engineering loss weights.
fe_color_rate: 0.03      # RGB soft histogram + color moments
fe_hog_rate: 0.02        # HOG/Sobel gradient orientation histogram
fe_dct_rate: 0.01        # low-frequency 2D DCT

# PCA-whitened attribute loss.
fe_pca_dim: 256
fe_pca_mix: 0.35
fe_pca_temperature: 10.0

# Runtime control.
fe_decode_every: 1       # set 2/4/8 if VRAM is tight
fe_warmup_steps: 0       # set 500~1000 if early training is unstable
```

## 第二次
100000 次训练，关闭 PCA 语义映射支持，FID 下降 1，其余不足，需要提升训练量

## 第三次
150000 次训练，关闭 PCA 语义，视觉特征增强 (0.6,0.4,0.2)，情感物体错误，emo_loss 明显提示 1.5 -> 2.5