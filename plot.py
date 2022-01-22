from matplotlib import pyplot as plt
import sys
import numpy as np
from collections import defaultdict
import pandas as pd
import os

transpose = {
    'Googlenet': 'output/coco/transpose_cv/TP_Googlenet_256x192_d256_h1024_enc4_mh8/TP_Googlenet_256x192_d256_h1024_enc4_mh8_2022-01-09-00-29_train.log',
    'InceptionV3': 'output/coco/transpose_cv/TP_InceptionV3_256x192_d256_h1024_enc4_mh8/TP_InceptionV3_256x192_d256_h1024_enc4_mh8_2022-01-09-00-29_train.log',
    'LinearProjection': 'output/coco/transpose_cv/TP_LinearProjection_256x192_d256_h1024_enc4_mh8/TP_LinearProjection_256x192_d256_h1024_enc4_mh8_2022-01-12-00-24_train.log',
    'MobileNetV1': 'output/coco/transpose_cv/TP_M_256x192_d256_h1024_enc3_mh8_M1/TP_M_256x192_d256_h1024_enc3_mh8_M1_2021-12-19-16-18_train.log',
    'MobileNetV2': 'output/coco/transpose_cv/TP_M_256x192_d256_h1024_enc3_mh8_M2/TP_M_256x192_d256_h1024_enc3_mh8_M2_2021-12-15-23-02_train.log',
    'MobileNetV3Small': 'output/coco/transpose_cv/TP_M_256x192_d256_h1024_enc3_mh8_M3L/TP_M_256x192_d256_h1024_enc3_mh8_M3L_2021-12-15-23-04_train.log',
    'MobileNetV3Large': 'output/coco/transpose_cv/TP_M_256x192_d256_h1024_enc3_mh8_M3S/TP_M_256x192_d256_h1024_enc3_mh8_M3S_2021-12-15-23-03_train.log',
    'ShuffleNetV2': 'output/coco/transpose_cv/TP_ShuffleNetV2_256x192_d256_h1024_enc4_mh8/TP_ShuffleNetV2_256x192_d256_h1024_enc4_mh8_2022-01-09-00-33_train.log',
    'SqueezeNet': 'output/coco/transpose_cv/TP_SqueezeNet_256x192_d256_h1024_enc4_mh8/TP_SqueezeNet_256x192_d256_h1024_enc4_mh8_2022-01-09-00-31_train.log',
    'Xception': 'output/coco/transpose_cv/TP_Xception_256x192_d256_h1024_enc4_mh8/TP_Xception_256x192_d256_h1024_enc4_mh8_2022-01-09-00-33_train.log',
    'HRNet': 'output/coco/transpose_h/TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc3_mh1/TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc3_mh1_2021-12-30-15-07_train.log',
    'ResNet50': 'output/coco/transpose_r/TP_R_256x192_d256_h1024_enc3_mh8/TP_R_256x192_d256_h1024_enc3_mh8_2021-12-30-15-05_train.log',
}

swin_transpose = {
    'Googlenet': 'output/coco/swin_transpose/STP_Googlenet_256x192_d256_h1024_enc4_mh8/STP_Googlenet_256x192_d256_h1024_enc4_mh8_2022-01-16-07-01_train.log',
    'InceptionV3': 'output/coco/swin_transpose/STP_InceptionV3_256x192_d256_h1024_enc4_mh8/STP_InceptionV3_256x192_d256_h1024_enc4_mh8_2022-01-15-09-56_train.log',
    'LinearProjection': 'output/coco/swin_transpose/STP_LinearProjection_256x192_d256_h1024_enc4_mh8/STP_LinearProjection_256x192_d256_h1024_enc4_mh8_2022-01-16-10-09_train.log',
    'MobileNetV1': 'output/coco/swin_transpose/STP_M_256x192_d256_h1024_enc3_mh8_M1/STP_M_256x192_d256_h1024_enc3_mh8_M1_2022-01-12-21-14_train.log',
    'MobileNetV2': 'output/coco/swin_transpose/STP_M_256x192_d256_h1024_enc3_mh8_M2/STP_M_256x192_d256_h1024_enc3_mh8_M2_2022-01-12-21-14_train.log',
    'MobileNetV3Small': 'output/coco/swin_transpose/STP_M_256x192_d256_h1024_enc3_mh8_M3S/STP_M_256x192_d256_h1024_enc3_mh8_M3S_2022-01-12-21-15_train.log',
    'MobileNetV3Large': 'output/coco/swin_transpose/STP_M_256x192_d256_h1024_enc3_mh8_M3L/STP_M_256x192_d256_h1024_enc3_mh8_M3L_2022-01-12-21-15_train.log',
    'ShuffleNetV2': 'output/coco/swin_transpose/STP_ShuffleNetV2_256x192_d256_h1024_enc4_mh8/STP_ShuffleNetV2_256x192_d256_h1024_enc4_mh8_2022-01-16-10-13_train.log',
    'SqueezeNet': 'output/coco/swin_transpose/STP_SqueezeNet_256x192_d256_h1024_enc4_mh8/STP_SqueezeNet_256x192_d256_h1024_enc4_mh8_2022-01-15-10-01_train.log',
    'Xception': 'output/coco/swin_transpose/STP_Xception_256x192_d256_h1024_enc4_mh8/STP_Xception_256x192_d256_h1024_enc4_mh8_2022-01-15-09-57_train.log',
    'HRNet': 'output/coco/swin_transposeH/STP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc3_mh1/STP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc3_mh1_2022-01-16-07-28_train.log',
    'ResNet50': 'output/coco/swin_transpose/STP_ResNet50_256x192_d256_h1024_enc4_mh8/STP_ResNet50_256x192_d256_h1024_enc4_mh8_2022-01-16-07-04_train.log',
}

tags = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
result = {}


def plot_figure(which):
    plt_tags = {
        'AP': 'b', 'AP (M)': 'r', 'AR': 'k', 'AR (M)': 'g'
    }
    plt_backbones = {
        'MobileNetV1': 'b', 
        'MobileNetV3Large': 'k', 
        'ResNet50': 'r', 
        'Xception': 'g', 
        'HRNet': 'grey', 
        'LinearProjection': 'orange'
    }
    save_dir = f'output/images/{which}'
    os.makedirs(save_dir, exist_ok=True)

    print(f'=========== begin plot {which}, save to {save_dir} ==============')
    for k in result:
        plt.figure()
        for tag in plt_tags:
            plt.plot(result[k][tag], label=tag, color=plt_tags[tag])
        plt.xlabel('epoch', fontsize=30)
        plt.xticks([0, 40, 80, 120], fontsize=30)
        plt.yticks(np.arange(0, 1, 0.2), fontsize=30)
        # plt.legend(fontsize=20)
        plt.savefig(f'{save_dir}/{k}.pdf', bbox_inches='tight', pad_inches=0)
    
    for tag in plt_tags:
        plt.figure()
        for k in plt_backbones:
            plt.plot(result[k][tag], label=k, color=plt_backbones[k])
        plt.xlabel('epoch', fontsize=30)
        plt.xticks([0, 40, 80, 120], fontsize=30)
        plt.yticks(np.arange(0, 1, 0.2), fontsize=30)
        # plt.legend(fontsize=20)
        plt.savefig(f'{save_dir}/{tag}.pdf', bbox_inches='tight', pad_inches=0)
    
    print(f'=========== finish plot {which}, save to {save_dir} ==============')
        

for which in ['transpose', 'swin_transpose']:
    result = {}
    if which == 'transpose':
        files = transpose
    else:
        files = swin_transpose
    for k, v in files.items():
        speeds = []
        d = defaultdict(list)
        with open(v) as f:
            for line in f:
                if all([i in line for i in ['Epoch', 'Time', 'Speed', 'samples/s	Data', 'Loss', 'Accuracy']]):
                    speeds.append(float(line.split('Speed')[1].split()[0].strip()))
                if f'| {v.split("/")[2]} |' in line and len(d[tags[0]]) < 120:
                    line = line.strip().split('|')[:-1]
                    for i in range(-1, -len(tags)-1, -1):
                        d[tags[i]].append(float(line[i].strip()))
        os.makedirs(f'output/data/{which}', exist_ok=True)
        pd.DataFrame(d).to_csv(f'output/data/{which}/{k}.csv', index=None)
        result[k] = d
        # print(k, f'{np.mean(speeds[1:]):.1f}', *[d[t][-1] for t in ['AP', 'AP (M)', 'AR', 'AR (M)']], sep=' & ')  # skip the first value to avoid cold start

    plot_figure(which)
    print()

tp_res = {}
for k, v in transpose.items():
    speeds = []
    d = defaultdict(list)
    with open(v) as f:
        for line in f:
            if all([i in line for i in ['Epoch', 'Time', 'Speed', 'samples/s	Data', 'Loss', 'Accuracy']]):
                speeds.append(float(line.split('Speed')[1].split()[0].strip()))
            if f'| {v.split("/")[2]} |' in line and len(d[tags[0]]) < 120:
                line = line.strip().split('|')[:-1]
                for i in range(-1, -len(tags)-1, -1):
                    d[tags[i]].append(float(line[i].strip()))
    d['speed'] = np.mean(speeds[1:])
    tp_res[k] = d

stp_res = {}
for k, v in swin_transpose.items():
    speeds = []
    d = defaultdict(list)
    with open(v) as f:
        for line in f:
            if all([i in line for i in ['Epoch', 'Time', 'Speed', 'samples/s	Data', 'Loss', 'Accuracy']]):
                speeds.append(float(line.split('Speed')[1].split()[0].strip()))
            if f'| {v.split("/")[2]} |' in line and len(d[tags[0]]) < 120:
                line = line.strip().split('|')[:-1]
                for i in range(-1, -len(tags)-1, -1):
                    d[tags[i]].append(float(line[i].strip()))
    d['speed'] = np.mean(speeds[1:])
    stp_res[k] = d

print(f'=========== begin draw table: impact of backbone ================')
for k in stp_res:
    tp, stp = tp_res[k], stp_res[k]
    print(k, f'{tp["speed"]:.1f}', *[f'{tp[t][-1]:.3f}' for t in ['AP', 'AP (M)', 'AR', 'AR (M)']], r'\\', sep=' & ')
print(f'=========== finish draw table: impact of backbone ================')
print()
print(f'=========== begin draw table: impact of transformer ================')
for k in stp_res:
    tp, stp = tp_res[k], stp_res[k]
    print(k, f'{stp["speed"]:.1f}({stp["speed"] - tp["speed"]:.1f})', *[f'{stp[t][-1]:.3f}({stp[t][-1]-tp[t][-1]:.3f})' for t in ['AP', 'AP (M)', 'AR', 'AR (M)']], r'\\', sep=' & ')
print(f'=========== finish draw table: impact of transformer ================')










