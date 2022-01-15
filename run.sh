CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg experiments/coco/transpose_cv/TP_Googlenet_256x192_d256_h1024_enc4_mh8.yaml
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg experiments/coco/transpose_cv/TP_InceptionV3_256x192_d256_h1024_enc4_mh8.yaml
CUDA_VISIBLE_DEVICES=2 python tools/train.py --cfg experiments/coco/transpose_cv/TP_SqueezeNet_256x192_d256_h1024_enc4_mh8.yaml

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg experiments/coco/transpose_cv/TP_Xception_256x192_d256_h1024_enc4_mh8.yaml
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg experiments/coco/transpose_cv/TP_ShuffleNetV2_256x192_d256_h1024_enc4_mh8.yaml
CUDA_VISIBLE_DEVICES=2 python tools/train.py --cfg experiments/coco/transpose_cv/TP_VGG16_256x192_d256_h1024_enc4_mh8.yaml


CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg experiments/coco/transpose_cv/TP_LinearProjection_256x192_d256_h1024_enc4_mh8.yaml
cd TransPose/ ; module load anaconda/3.7.1 ; source activate cv

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg experiments/coco/swin_transpose/STP_Googlenet_256x192_d256_h1024_enc4_mh8.yaml
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg experiments/coco/swin_transpose/STP_InceptionV3_256x192_d256_h1024_enc4_mh8.yaml
CUDA_VISIBLE_DEVICES=2 python tools/train.py --cfg experiments/coco/swin_transpose/STP_LinearProjection_256x192_d256_h1024_enc4_mh8.yaml


CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg experiments/coco/swin_transpose/STP_M_256x192_d256_h1024_enc3_mh8_M1.yaml
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg experiments/coco/swin_transpose/STP_M_256x192_d256_h1024_enc3_mh8_M2.yaml
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg experiments/coco/swin_transpose/STP_M_256x192_d256_h1024_enc3_mh8_M3L.yaml
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg experiments/coco/swin_transpose/STP_M_256x192_d256_h1024_enc3_mh8_M3S.yaml

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg experiments/coco/swin_transpose/STP_ShuffleNetV2_256x192_d256_h1024_enc4_mh8.yaml
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg experiments/coco/swin_transpose/STP_SqueezeNet_256x192_d256_h1024_enc4_mh8.yaml
CUDA_VISIBLE_DEVICES=2 python tools/train.py --cfg experiments/coco/swin_transpose/STP_Xception_256x192_d256_h1024_enc4_mh8.yaml
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg experiments/coco/swin_transpose/STP_ResNet50_256x192_d256_h1024_enc4_mh8.yaml