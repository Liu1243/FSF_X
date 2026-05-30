# FSF_HierMamba_nuScenes 训练结果汇总

更新时间：2026-05-23  
用途：PPT 汇报材料整理  
模型：FSF_HierMamba_nuScenes

## 1. 汇报结论

- mini 数据集上完成了多组 6 epoch 验证实验，最优版本为 `FSF_HierMamba_nuScenes_mini_config_opt`，NDS = 55.24，mAP = 52.84。该结果主要用于验证训练链路和模型改动有效性；由于 mini 类别分布不完整，trailer、construction_vehicle、barrier 等类别 AP 为 0，不宜作为最终性能判断。
- 60% nuScenes 主实验 `FSF_HierMamba_nuScenes_60pct_fp32_stable` 已完成 6 epoch，并完成 epoch 1-6 的逐轮评估。最优 NDS 出现在 epoch 6，NDS = 65.64；最优 mAP 出现在 epoch 4，mAP = 59.22。epoch 4-6 已进入平台期，NDS 仍有小幅提升。
- 全量 nuScenes 主实验 `FSF_HierMamba_nuScenes_full_fp32_stable` 已完成 4 个完整 epoch 的 checkpoint，第 5 个 epoch 有部分训练日志，但尚未看到对应 checkpoint 和验证指标。当前只能汇报训练状态和 loss 趋势，不能与 mini/60% 的 NDS、mAP 直接比较。
- 建议 PPT 主线：mini 证明模型可训练，60% 数据集证明性能稳定提升，全量数据集展示正在扩大规模训练；后续补全全量数据集评估后再作为最终结果页。

## 2. 实验概览

| 数据设置 | 主要实验目录 | 训练轮次 | 评估状态 | 核心结论 |
| --- | --- | ---: | --- | --- |
| mini | `work_dirs/nuScenes/FSF_HierMamba_nuScenes_mini_config*` | 6 epoch | 有最终评估 | 最优 NDS 55.24，mAP 52.84；适合作为链路验证结果 |
| 60% nuScenes | `work_dirs/nuScenes/FSF_HierMamba_nuScenes_60pct_fp32_stable` | 6 epoch | 有 epoch 1-6 评估 | 最优 NDS 65.64，mAP 59.22；当前最适合 PPT 展示 |
| full nuScenes | `work_dirs/nuScenes/FSF_HierMamba_nuScenes_full_fp32_stable` | 完整 checkpoint 到 epoch 4，epoch 5 部分日志 | 暂无评估指标 | 训练 loss 稳定下降，需要补跑验证 |

## 3. mini 数据集结果

配置要点：

- 数据根目录：`data/nuscenes_mini/`
- 本地 pkl 样本数：train 323，val 81
- 训练配置：`max_epochs=6`，`samples_per_gpu=1`，`sweeps_num=9`
- 评估指标：nuScenes detection NDS / mAP

mini 数据集多组实验对比：

| 实验 | epoch | NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `mini_config` | 6 | 54.24 | 51.12 | 0.4259 | 0.4686 | 0.5493 | 0.3981 | 0.2902 | 初始稳定版本 |
| `mini_config_opt` | 6 | **55.24** | **52.84** | **0.4165** | **0.4637** | **0.5372** | **0.3978** | 0.3027 | mini 最优版本 |
| `mini_config_opt2` | 6 | 53.35 | 50.82 | 0.4234 | 0.4730 | 0.6065 | 0.4056 | 0.2981 | 优化方向未超过 opt |

mini 最优版本类别 AP：

| 类别 | AP |
| --- | ---: |
| car | 87.29 |
| truck | 78.39 |
| bus | 94.55 |
| pedestrian | 90.40 |
| motorcycle | 67.89 |
| bicycle | 34.94 |
| traffic_cone | 74.92 |
| trailer | 0.00 |
| construction_vehicle | 0.00 |
| barrier | 0.00 |

解读：

- `mini_config_opt` 相比初始版本 NDS 提升 1.00 个百分点，mAP 提升 1.72 个百分点。
- car、bus、pedestrian、traffic_cone 在 mini 上表现较好，说明主干检测链路有效。
- trailer、construction_vehicle、barrier 为 0，主要受 mini 数据规模和类别覆盖影响，PPT 中应强调 mini 仅用于快速验证。

## 4. 60% nuScenes 结果

配置要点：

- 训练 ann 文件：`data/nuscenes/nuscenes_infos_train_60pct.pkl`
- 验证 ann 文件：`data/nuscenes/nuscenes_infos_val.pkl`
- 训练配置：`max_epochs=6`，`samples_per_gpu=1`，`workers_per_gpu=4`
- 主实验学习率：`lr=2e-05`
- checkpoint：`epoch_1.pth` 到 `epoch_6.pth` 均存在

逐 epoch 评估结果：

| epoch | NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 63.21 | 57.77 | 0.2093 | 0.1931 | 0.2872 | 0.6699 | 0.2082 |
| 2 | 64.25 | 58.16 | 0.2104 | 0.1909 | 0.2796 | 0.5953 | 0.2063 |
| 3 | 65.04 | 58.80 | 0.2043 | 0.1874 | 0.2678 | 0.5712 | 0.2056 |
| 4 | 65.34 | **59.22** | 0.2035 | 0.1858 | 0.2666 | 0.5704 | 0.2013 |
| 5 | 65.63 | 59.16 | 0.2003 | 0.1830 | 0.2657 | **0.5430** | 0.2036 |
| 6 | **65.64** | 59.11 | **0.2000** | **0.1814** | **0.2524** | 0.5562 | **0.2011** |

60% 结果解读：

- epoch 1 到 epoch 6，NDS 从 63.21 提升到 65.64，提升 2.43 个百分点。
- mAP 从 57.77 提升到 59.11，提升 1.34 个百分点；最高值在 epoch 4，为 59.22。
- 定位、尺度、朝向误差持续下降：mATE 0.2093 -> 0.2000，mASE 0.1931 -> 0.1814，mAOE 0.2872 -> 0.2524。
- epoch 4-6 指标接近平台期，说明 60% 数据集上 6 epoch 已经得到较稳定结果；如果继续训练，需要关注是否能带来 mAP 的有效提升。

epoch 6 类别 AP：

| 类别 | AP |
| --- | ---: |
| bus | 78.51 |
| car | 67.65 |
| truck | 64.88 |
| barrier | 58.19 |
| bicycle | 57.66 |
| motorcycle | 57.51 |
| trailer | 54.48 |
| pedestrian | 54.48 |
| construction_vehicle | 51.96 |
| traffic_cone | 45.80 |

和 mini 最优结果对比：

| 数据设置 | NDS | mAP | 说明 |
| --- | ---: | ---: | --- |
| mini best | 55.24 | 52.84 | 类别覆盖不足，适合验证训练链路 |
| 60% best NDS | **65.64** | 59.11 | 类别覆盖完整，NDS 最优 |
| 60% best mAP | 65.34 | **59.22** | mAP 最优，epoch 4 |

## 5. full nuScenes 训练状态

配置要点：

- 训练 ann 文件：`data/nuscenes/nuscenes_infos_train.pkl`
- 验证 ann 文件：`data/nuscenes/nuscenes_infos_val.pkl`
- 训练配置：`max_epochs=6`，`samples_per_gpu=1`，`workers_per_gpu=4`
- 主实验学习率：`lr=2e-05`
- checkpoint：当前看到 `epoch_1.pth` 到 `epoch_4.pth`，以及 `latest.pth`

训练日志状态：

| 阶段 | 训练进度 | 末次记录 loss | 备注 |
| --- | ---: | ---: | --- |
| epoch 1 | 128100 / 128100 iter | 2.3496 | 完整 epoch |
| epoch 2 | 128100 / 128100 iter | 2.3980 | 完整 epoch |
| epoch 3 | 128100 / 128100 iter | 2.1260 | 完整 epoch |
| epoch 4 | 128100 / 128100 iter | 2.0725 | 完整 epoch，已有 checkpoint |
| epoch 5 | 24820 / 128100 iter | 1.9877 | 部分训练日志，未看到 epoch 5 checkpoint |

解读：

- 全量训练从 `ckpt/fsd_nusc_pretrain.pth` 初始化，日志显示已进入 full 数据规模训练。
- 训练 loss 从 epoch 1 的 2.3496 降到 epoch 4 的 2.0725，第 5 轮部分日志降到 1.9877，整体趋势正常。
- 当前缺少 full 数据集的评估结果，PPT 中应标注为“训练进行中/待验证”，不要和 60% 的 NDS、mAP 做直接性能对比。

## 6. PPT 页面建议

1. 实验背景页：FSF_HierMamba 在 nuScenes 上的规模化训练进展，分 mini、60%、full 三个阶段。
2. 实验设置页：列出三类数据设置、训练轮次、checkpoint 状态、是否已评估。
3. mini 结果页：展示 best NDS 55.24、mAP 52.84，说明用于快速验证和调参。
4. 60% 结果页：用逐 epoch 表或折线图突出 NDS 从 63.21 到 65.64，mAP 在 59.1 左右稳定。
5. 类别 AP 页：展示 60% epoch 6 类别 AP，说明完整类别覆盖后 rare classes 不再为 0。
6. full 训练状态页：展示训练已完成 4 个完整 epoch，第 5 轮部分进行，强调下一步补全评估。
7. 结论与计划页：60% 结果已经可作为当前核心性能汇报，full 结果需要补跑 validation 后更新最终结论。

## 7. 建议下一步

- 对 full nuScenes 的 `epoch_4.pth` 先跑一次 validation，补齐 NDS、mAP 和类别 AP。
- 如果算力允许，继续完成 full 的 epoch 5-6，并分别评估，判断是否出现和 60% 一样的平台期。
- 60% 实验可以补充曲线图：NDS、mAP、mATE、mAOE 四条趋势最适合 PPT 展示。
- 对 mini 实验只保留 best 版本和一句限制说明，避免听众误解 rare class AP 为 0 是模型能力问题。

## 8. 数据来源

| 内容 | 来源文件 |
| --- | --- |
| mini 初始版本评估 | `work_dirs/nuScenes/FSF_HierMamba_nuScenes_mini_config/eval/results/metrics_summary.json` |
| mini opt 评估 | `work_dirs/nuScenes/FSF_HierMamba_nuScenes_mini_config_opt/eval/results/metrics_summary.json` |
| mini opt2 评估 | `work_dirs/nuScenes/FSF_HierMamba_nuScenes_mini_config_opt2/eval/results/metrics_summary.json` |
| 60% epoch 1-6 评估 | `work_dirs/nuScenes/FSF_HierMamba_nuScenes_60pct_fp32_stable/eval_train60/epoch_*/metrics_summary.json` |
| 60% 训练日志 | `work_dirs/nuScenes/FSF_HierMamba_nuScenes_60pct_fp32_stable/*.log.json` |
| full 训练日志 | `work_dirs/nuScenes/FSF_HierMamba_nuScenes_full_fp32_stable/*.log.json` |
| full checkpoint 状态 | `work_dirs/nuScenes/FSF_HierMamba_nuScenes_full_fp32_stable/epoch_*.pth` |

