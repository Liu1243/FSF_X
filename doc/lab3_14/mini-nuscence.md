## FSF
mAP: 0.4970
mATE: 0.4245
mASE: 0.4676
mAOE: 0.6629
mAVE: 0.4321
mAAE: 0.2910
NDS: 0.5207
Eval time: 2.3s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.867   0.182   0.157   0.136   0.122   0.087
truck   0.765   0.187   0.145   0.091   0.115   0.006
bus     0.960   0.160   0.093   0.052   0.640   0.055
trailer 0.000   1.000   1.000   1.000   1.000   1.000
construction_vehicle    0.000   1.000   1.000   1.000   1.000   1.000
pedestrian      0.913   0.117   0.276   0.263   0.193   0.132
motorcycle      0.700   0.243   0.384   0.793   0.076   0.014
bicycle 0.102   0.293   0.217   1.631   0.311   0.033
traffic_cone    0.662   0.064   0.402   nan     nan     nan
barrier 0.000   1.000   1.000   1.000   nan     nan

## H-Mamba
### train
PYTHONPATH="$(pwd)":$PYTHONPATH python tools/train.py \
    projects/configs/nuScenes/FSF_HMamba_nuScenes_config.py \
    --work-dir work_dirs/nuScenes/FSF_HMamba_nuScenes_config \
    --cfg-options evaluation.jsonfile_prefix=work_dirs/nuScenes/FSF_HMamba_nuScenes_config/eval/results

### test
单卡测试
PYTHONPATH="$(pwd)":$PYTHONPATH python tools/test.py \
    projects/configs/nuScenes/FSF_HMamba_nuScenes_config.py \
    work_dirs/nuScenes/FSF_HMamba_nuScenes_config/epoch_6.pth \
    --eval bbox

mini-nuscenes
mAP: 0.5060
mATE: 0.4244
mASE: 0.4630
mAOE: 0.6657
mAVE: 0.4230
mAAE: 0.2901
NDS: 0.5264
Eval time: 2.7s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.866   0.183   0.162   0.147   0.139   0.093
truck   0.761   0.195   0.154   0.081   0.095   0.000
bus     0.989   0.136   0.106   0.030   0.599   0.059
trailer 0.000   1.000   1.000   1.000   1.000   1.000
construction_vehicle    0.000   1.000   1.000   1.000   1.000   1.000
pedestrian      0.910   0.111   0.275   0.276   0.200   0.126
motorcycle      0.685   0.234   0.358   0.872   0.072   0.020
bicycle 0.134   0.319   0.215   1.586   0.279   0.023
traffic_cone    0.714   0.065   0.359   nan     nan     nan
barrier 0.000   1.000   1.000   1.000   nan     nan

### 展示指标
mAP / NDS → 最终性能提升
小目标类别 AP（pedestrian, bicycle, traffic_cone）→ 体现稀疏点云场景的改善
mATE（定位误差） → 体现多模态交互对空间感知的提升
推理速度（FPS 或 latency） → 若 Mamba 相比 Transformer self-attention 更快

## Occ
### test
PYTHONPATH="$(pwd)":$PYTHONPATH python tools/test.py \
    projects/configs/nuScenes/FSF_Occ_nuScenes_mini_config.py \
    work_dirs/nuScenes/FSF_Occ_nuScenes_mini_config/epoch_1.pth \
    --eval bbox

mAP: 0.5659
mATE: 0.4190
mASE: 0.4613
mAOE: 0.6285
mAVE: 0.4747
mAAE: 0.2877
NDS: 0.5558
Eval time: 2.5s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.873   0.178   0.154   0.156   0.102   0.080
truck   0.846   0.179   0.174   0.099   0.105   0.000
bus     0.989   0.188   0.120   0.039   0.959   0.078
trailer 0.000   1.000   1.000   1.000   1.000   1.000
construction_vehicle    0.000   1.000   1.000   1.000   1.000   1.000
pedestrian      0.905   0.117   0.238   0.257   0.177   0.144
motorcycle      0.726   0.256   0.333   1.068   0.055   0.000
bicycle 0.526   0.206   0.196   1.037   0.399   0.000
traffic_cone    0.794   0.067   0.397   nan     nan     nan
barrier 0.000   1.000   1.000   1.000   nan     nan