CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg experiments/coco/transpose_cv/TP_Googlenet_256x192_d256_h1024_enc4_mh8.yaml
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg experiments/coco/transpose_cv/TP_InceptionV3_256x192_d256_h1024_enc4_mh8.yaml
CUDA_VISIBLE_DEVICES=2 python tools/train.py --cfg experiments/coco/transpose_cv/TP_SqueezeNet_256x192_d256_h1024_enc4_mh8.yaml
CUDA_VISIBLE_DEVICES=3 python tools/train.py --cfg experiments/coco/transpose_cv/TP_VGG16_256x192_d256_h1024_enc4_mh8.yaml
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg experiments/coco/transpose_cv/TP_Xception_256x192_d256_h1024_enc4_mh8.yaml
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg experiments/coco/transpose_cv/TP_ShuffleNetV2_256x192_d256_h1024_enc4_mh8.yaml
CUDA_VISIBLE_DEVICES=2 python tools/train.py --cfg experiments/coco/transpose_cv/TP_LinearProjection_256x192_d256_h1024_enc4_mh8.yaml
