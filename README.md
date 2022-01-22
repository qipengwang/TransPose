## Introduction

This repo contains codes for CV final project: **When Transformer Meets Pose Estimation: Methodology, Measurement, and Analysis**. It is based on **[TransPose](https://arxiv.org/abs/2012.14214)**, which is a human pose estimation model based on a CNN feature extractor, a Transformer Encoder, and a prediction head. Given an image, the attention layers built in Transformer can efficiently capture long-range spatial relationships between keypoints and explain what dependencies the predicted keypoints locations highly rely on. 

![Architecture](transpose_architecture.png)



## Experiment results

Here is all of the experiment result of our project.

All of the experiments are conducted on COCO2017 dataset.

All experiments were done on 2 nodes of the [PKU PHC platform](https://hpc.pku.edu.cn), each with 4 Telsa P100 GPUs. 



### Results using basic Transformer

Table1 in report

| Backbone  | Throughput (fps) | AP | AP (M) | AR | AR (m) |
| -------------- | ----: | :---------------: | :--: | :---------------: | :--: |
| Googlenet | 87.7 | 0.701 | 0.672 | 0.733 | 0.700 |
| InceptionV3 | 81.1 | 0.721 | 0.693 | 0.751 | 0.719 |
| LinearProjection | 100.9 | 0.361 | 0.353 | 0.408 | 0.388 |
| MobileNetV1 | 114.7 | 0.670 | 0.643 | 0.703 | 0.670 |
| MobileNetV2 | 108.8 | 0.649 | 0.621 | 0.683 | 0.650 |
| MobileNetV3Small | 112.2 | 0.642 | 0.614 | 0.677 | 0.643 |
| MobileNetV3Large | 123.5 | 0.583 | 0.562 | 0.621 | 0.594 |
| ShuffleNetV2 | 96.2 | 0.659 | 0.635 | 0.692 | 0.662 |
| SqueezeNet | 83.6 | 0.695 | 0.666 | 0.726 | 0.693 |
| Xception | 77.6 | 0.702 | 0.673 | 0.732 | 0.698 |
| HRNet | 39.4 | 0.751 | 0.723 | 0.780 | 0.749 |
| ResNet50 | 95.2 | 0.711 | 0.681 | 0.741 | 0.705 |



### Results using Swin-Transformer

Table2 in report

| Backbone  | Throughput (fps) | AP | AP (M) | AR | AR (m) |
| -------------- | ----: | :---------------: | :--: | :---------------: | :--: |
| Googlenet | 62.3(-25.4) | 0.503(-0.198) | 0.469(-0.203) | 0.554(-0.179) | 0.512(-0.188) |
| InceptionV3 | 59.0(-22.1) | 0.541(-0.180) | 0.509(-0.184) | 0.586(-0.165) | 0.547(-0.172) |
| LinearProjection | 68.3(-32.5) | 0.100(-0.261) | 0.104(-0.249) | 0.168(-0.240) | 0.151(-0.237) |
| MobileNetV1 | 67.8(-46.9) | 0.473(-0.197) | 0.437(-0.206) | 0.524(-0.179) | 0.483(-0.187) |
| MobileNetV2 | 66.2(-42.5) | 0.469(-0.180) | 0.432(-0.189) | 0.519(-0.164) | 0.475(-0.175) |
| MobileNetV3Small | 71.9(-40.3) | 0.387(-0.255) | 0.367(-0.247) | 0.445(-0.232) | 0.414(-0.229) |
| MobileNetV3Large | 67.9(-55.6) | 0.521(-0.062) | 0.496(-0.066) | 0.565(-0.056) | 0.532(-0.062) |
| ShuffleNetV2 | 66.4(-29.8) | 0.401(-0.258) | 0.375(-0.260) | 0.457(-0.235) | 0.422(-0.240) |
| SqueezeNet | 60.5(-23.1) | 0.510(-0.185) | 0.478(-0.188) | 0.559(-0.167) | 0.519(-0.174) |
| Xception | 57.3(-20.3) | 0.514(-0.188) | 0.475(-0.198) | 0.564(-0.168) | 0.517(-0.181) |
| HRNet | 24.9(-14.5) | 0.750(-0.001) | 0.722(-0.001) | 0.778(-0.002) | 0.746(-0.003) |
| ResNet50 | 56.7(-38.5) | 0.566(-0.145) | 0.529(-0.152) | 0.608(-0.133) | 0.567(-0.138) |




## Getting started

### Installation

Please clone this repo and follow the [TransPose Official Repo](https://github.com/yangsenius/TransPose)'s instructions to build the project.

1. Clone this repository, and we'll call the directory that you cloned as ${POSE_ROOT}

   ```bash
   git clone https://github.com/qipengwang/TransPose.git
   ```



### Running 

```bash
[CUDA_VISIBLE_DEVICES=0] python tools/train.py --cfg {CONFIG_PATH}
```

Optionally to set the `CUDA_VISIBLE_DEVICES` environment variable. This command uses **one** GPU by default.

The `{CONFIG_PATH}` is the `yaml` file path to config the training process.

The `{CONFIG_PATH}` we used in our experiment is as following. 

**Note** that Transformer with ResNet50/HRNet is our **reproduced** result of original TransPose.

The default output directory is `output/{DATASET}/{CONFIG.MODEL.NAME}/{CONFIG_FILE_NOEXTENSION}/`, e.g., `output/coco/transpose_h/TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc3_mh1/`.

| Backbone         | Transformer                                                  | Swin-Transformer                                             |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ResNet50         | *[ResNet50 + Transformer](experiments/coco//transpose_r/TP_R_256x192_d256_h1024_enc3_mh8.yaml)* | [ResNet50+Swin-Transformer](experiments/coco/swin_transpose/STP_ResNet50_256x192_d256_h1024_enc4_mh8.yaml) |
| HRNet            | *[HRNet+Transformer](experiments/coco/transpose_h/TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc3_mh1.yaml)* | [HRNet+Swin-Transformer](experiments/coco/STP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc3_mh1.yaml) |
| MobileNetV1      | [MobileNetV1 + Transformer](experiments/coco/transpose_cv/TP_M_256x192_d256_h1024_enc3_mh8_M1.yaml) | [MobileNetV1+Swin-Transformer](experiments/coco/STP_M_256x192_d256_h1024_enc3_mh8_M1.yaml) |
| MobileNetV2      | [MobileNetV2+ Transformer](experiments/coco/transpose_cv/TP_M_256x192_d256_h1024_enc3_mh8_M2.yaml) | [MobileNetV2+Swin-Transformer](experiments/coco/STP_M_256x192_d256_h1024_enc3_mh8_M2.yaml) |
| MobileNetV3Small | [MobileNetV3Small+ Transformer](experiments/coco/transpose_cv/TP_M_256x192_d256_h1024_enc3_mh8_M3S.yaml) | [MobileNetV3Small+Swin-Transformer](experiments/coco/STP_M_256x192_d256_h1024_enc3_mh8_M3S.yaml) |
| MobileNetV3Large | [MobileNetV3Large + Transformer](experiments/coco/transpose_cv/TP_M_256x192_d256_h1024_enc3_mh8_M3L.yaml) | [MobileNetV3Large+Swin-Transformer](experiments/coco/STP_M_256x192_d256_h1024_enc3_mh8_M3L.yaml) |
| ShuffleNetV2     | [ShuffleNetV2 + Transformer](experiments/coco/transpose_cv/TP_ShuffleNetV2_256x192_d256_h1024_enc4_mh8.yaml) | [ShuffleNetV2+Swin-Transformer](experiments/coco/STP_ShuffleNetV2_256x192_d256_h1024_enc4_mh8.yaml) |
| InceptionV3      | [InceptionV3 + Transformer](experiments/coco/transpose_cv/TP_InceptionV3_256x192_d256_h1024_enc4_mh8.yaml) | [InceptionV3+Swin-Transformer](experiments/coco/STP_InceptionV3_256x192_d256_h1024_enc4_mh8.yaml) |
| GoogleNet        | [GoogleNet + Transformer](experiments/coco/transpose_cv/TP_Googlenet_256x192_d256_h1024_enc4_mh8.yaml) | [GoogleNet+Swin-Transformer](experiments/coco/STP_Googlenet_256x192_d256_h1024_enc4_mh8.yaml) |
| SqueezeNet       | [SqueezeNet + Transformer](experiments/coco/transpose_cv/TP_SqueezeNet_256x192_d256_h1024_enc4_mh8.yaml) | [SqueezeNet+Swin-Transformer](experiments/coco/STP_SqueezeNet_256x192_d256_h1024_enc4_mh8.yaml) |
| Xception         | [Xception + Transformer](experiments/coco/transpose_cv/TP_Xception_256x192_d256_h1024_enc4_mh8.yaml) | [Xception+Swin-Transformer](experiments/coco/STP_Xception_256x192_d256_h1024_enc4_mh8.yaml) |
| LinearProjection | [LinearProjection + Transformer](experiments/coco/transpose_cv/TP_LinearProjection_256x192_d256_h1024_enc4_mh8.yaml) | [LinearProjection+Swin-Transformer](experiments/coco/STP_LinearProjection_256x192_d256_h1024_enc4_mh8.yaml) |



### Plot figures and tables

to plot the figures and tables in report, please run the following command in the `{REPO_ROOT}` directory:

```shell
unzip output.zip
python plot.py
```

`output.zip` contains all of the training log of our experiments, as well as the figures and processed `csv` files.



### Acknowledgements

Great thanks for [TransPose](https://arxiv.org/pdf/2012.14214.pdf) and their open-source [code](https://github.com/yangsenius/TransPose). Thanks for the open-sourced [SwinTransformer](https://github.com/microsoft/Swin-Transformer) and all of the backbones.



## Members and contributions

**ALL of the members contributes equally to this project!**

- [Zeyu Yang](https://github.com/yzy-pku)
- [Jinwei Chen](https://github.com/WayneChen-cloud)
- [Diandian Gu](https://github.com/gidiandian)
- [Qipeng Wang](https://github.com/qipengwang)

## License

This repository is released under the [MIT LICENSE](https://github.com/yangsenius/TransPose/blob/main/LICENSE).
