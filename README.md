## Introduction

This repo contains codes for CV final project: **When Transformer Meets Pose Estimation: Methodology, Measurement, and Analysis**. It is based on **[TransPose](https://arxiv.org/abs/2012.14214)**, which is a human pose estimation model based on a CNN feature extractor, a Transformer Encoder, and a prediction head. Given an image, the attention layers built in Transformer can efficiently capture long-range spatial relationships between keypoints and explain what dependencies the predicted keypoints locations highly rely on. 

![Architecture](transpose_architecture.png)



## Experiment results

Here is all of the experiment result of our project.

All of the experiments are conducted on COCO2017 dataset.

All experiments were done on 2 nodes of the [PKU PHC platform](https://hpc.pku.edu.cn), each with 4 Telsa P100 GPUs. 



### Results using basic Transformer

| Backbone  | Throughput (fps) | mAP | mAR |
| -------------- | :---------: | :---------------: | :--: |
| ResNet50         |                  |      |  |
| HRNet            |                  |      |  |
| MobileNetV1      |                  |      |  |
| MobileNetV2      |                  |      |  |
| MobileNetV3Small |                  |      |  |
| MobileNetV3Large |                  |      |  |
| ShuffleNetV2     |                  |      |  |
| InceptionV3      |                  |      |  |
| GoogleNet        |                  |      |  |



### Results using Swin-Transformer

| Backbone         | Throughput (fps) | mAP  | mAR  |
| ---------------- | :--------------: | :--: | :--: |
| ResNet50         |                  |      |      |
| HRNet            |                  |      |      |
| MobileNetV1      |                  |      |      |
| MobileNetV2      |                  |      |      |
| MobileNetV3Small |                  |      |      |
| MobileNetV3Large |                  |      |      |
| ShuffleNetV2     |                  |      |      |
| InceptionV3      |                  |      |      |
| GoogleNet        |                  |      |      |




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

| Backbone         | Transformer (relative path to experiments/coco/)             | Swin-Transformer (relative path to experiments/coco/swin_transpose) |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ResNet50         | */transpose_r/TP_R_256x192_d256_h1024_enc3_mh8.yaml*         | STP_ResNet50_256x192_d256_h1024_enc4_mh8.yaml                |
| HRNet            | */transpose_h/TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc3_mh1.yaml* | STP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc3_mh1.yaml     |
| MobileNetV1      | /transpose_cv/TP_M_256x192_d256_h1024_enc3_mh8_M1.yaml       | STP_M_256x192_d256_h1024_enc3_mh8_M1.yaml                    |
| MobileNetV2      | /transpose_cv/TP_M_256x192_d256_h1024_enc3_mh8_M2.yaml       | STP_M_256x192_d256_h1024_enc3_mh8_M2.yaml                    |
| MobileNetV3Small | /transpose_cv/TP_M_256x192_d256_h1024_enc3_mh8_M3S.yaml      | STP_M_256x192_d256_h1024_enc3_mh8_M3S.yaml                   |
| MobileNetV3Large | /transpose_cv/TP_M_256x192_d256_h1024_enc3_mh8_M3L.yaml      | STP_M_256x192_d256_h1024_enc3_mh8_M3L.yaml                   |
| ShuffleNetV2     | /transpose_cv/TP_ShuffleNetV2_256x192_d256_h1024_enc4_mh8.yaml | STP_ShuffleNetV2_256x192_d256_h1024_enc4_mh8.yaml            |
| InceptionV3      | /transpose_cv/TP_InceptionV3_256x192_d256_h1024_enc4_mh8.yaml | STP_InceptionV3_256x192_d256_h1024_enc4_mh8.yaml             |
| GoogleNet        | /transpose_cv/TP_Googlenet_256x192_d256_h1024_enc4_mh8.yaml  | STP_Googlenet_256x192_d256_h1024_enc4_mh8.yaml               |
| SqueezeNet       | /transpose_cv/TP_SqueezeNet_256x192_d256_h1024_enc4_mh8.yaml | STP_SqueezeNet_256x192_d256_h1024_enc4_mh8.yaml              |
| Xception         | /transpose_cv/TP_Xception_256x192_d256_h1024_enc4_mh8.yaml   | STP_Xception_256x192_d256_h1024_enc4_mh8.yaml                |
| LinearProjection | /transpose_cv/TP_LinearProjection_256x192_d256_h1024_enc4_mh8.yaml | STP_LinearProjection_256x192_d256_h1024_enc4_mh8.yaml        |



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
