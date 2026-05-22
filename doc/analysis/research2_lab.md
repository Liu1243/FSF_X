# Research 2 Lab: FSF_Occ nuScenes Mini 实验记录

本文档记录研究内容2 `FSF_Occ` 在 nuScenes mini 上的实验结果、问题分析和后续改进。

## 记录说明

- 数据集：`data/nuscenes_mini/`
- 配置：`projects/configs/nuScenes/FSF_Occ_nuScenes_mini_config.py`
- 主要指标：nuScenes `mAP`、`NDS` 和 TP error 指标 `mATE/mASE/mAOE/mAVE/mAAE`
- mini split 类别覆盖不完整。当前 `val` 统计中有有效 GT 的检测类为 `car/truck/bus/bicycle/motorcycle/pedestrian/traffic_cone`，`trailer/construction_vehicle/barrier` 在该 split 中没有有效 GT，因此官方 10 类 mAP 会被这三个 0 AP 类别显著拉低。
- 为辅助判断真实改进，本文同时记录 `present-class mAP`，即只对有有效 val GT 的 7 个类别计算 AP 均值。

## mini 类别分布

统计命令：

```bash
wsl --cd /home/ddd/pc/FullySparseFusion bash -lc "source /home/ddd/miniconda3/etc/profile.d/conda.sh && conda activate FSF && python - <<'PY'
import pickle, collections
for split in ['train','val']:
    path=f'data/nuscenes_mini/nuscenes_infos_{split}.pkl'
    with open(path,'rb') as f:
        data=pickle.load(f)
    infos=data['infos'] if isinstance(data, dict) and 'infos' in data else data
    c=collections.Counter()
    for info in infos:
        c.update(map(str, info.get('gt_names', [])))
    print(split, len(infos), dict(c))
PY"
```

| class | train GT | val GT | 观察 |
| --- | ---: | ---: | --- |
| car | 5051 | 2568 | 高频，指标稳定 |
| truck | 525 | 124 | 中频，表现较好 |
| bus | 369 | 41 | val 样本少，AP 波动风险高 |
| trailer | 60 | 0 | val 无有效 GT，官方 AP 计 0 |
| construction_vehicle | 196 | 0 | val 无有效 GT，官方 AP 计 0 |
| bicycle | 191 | 52 | 低频小目标，是主要短板之一 |
| motorcycle | 212 | 259 | 样本可用，但朝向误差偏高 |
| pedestrian | 3657 | 1358 | 高频，指标稳定 |
| traffic_cone | 1339 | 39 | 小目标，scale error 偏高 |
| barrier | 2323 | 0 | train 有大量样本但 val 无有效 GT，官方 AP 计 0 |

## 实验总览

| 版本 | 结果目录 | 主要改动 | mAP | NDS | present-class mAP | 状态 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| FSF baseline | `work_dirs/nuScenes/FSF_nuScenes_mini_config` | 原始 FSF mini 配置 | 0.4970 | 0.5207 | 0.7099 | 已完成 |
| FSF_Occ opt1 | `work_dirs/nuScenes/FSF_Occ_nuScenes_mini_config_opt1` | coarse-to-fine occupancy、实例级 completion descriptor、size/visibility 辅助损失 | 0.5089 | 0.5353 | 0.7270 | 已完成 |
| FSF_Occ opt2 | `work_dirs/nuScenes/FSF_Occ_nuScenes_mini_config_opt2` | 稀疏实例 top-K 保点，`min_points_per_instance=2` | 0.5106 | 0.5358 | 0.7296 | 已完成 |

## 实验 1: FSF_Occ opt1

结果来源：用户提供日志，训练输出位于 `work_dirs/nuScenes/FSF_Occ_nuScenes_mini_config_opt1/20260505_225901.log.json`。最佳 checkpoint 为 `best_pts_bbox_NuScenes/NDS_epoch_6.pth`，最佳 NDS 出现在 epoch 6。

### 整体指标

| mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5089 | 0.5353 | 0.4253 | 0.4709 | 0.5844 | 0.4220 | 0.2893 |

### 分类别结果

| class | AP | ATE | ASE | AOE | AVE | AAE | 观察 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| car | 0.865 | 0.186 | 0.157 | 0.150 | 0.121 | 0.083 | 高频类稳定，定位和朝向较好 |
| truck | 0.755 | 0.201 | 0.162 | 0.078 | 0.151 | 0.031 | 表现较好，但较 FSF baseline 略低 |
| bus | 0.962 | 0.170 | 0.128 | 0.044 | 0.540 | 0.061 | AP 很高，速度误差偏大 |
| trailer | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | mini val 无有效 GT，官方 AP 计 0 |
| construction_vehicle | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | mini val 无有效 GT，官方 AP 计 0 |
| pedestrian | 0.904 | 0.123 | 0.285 | 0.247 | 0.202 | 0.126 | 行人检测稳定 |
| motorcycle | 0.721 | 0.229 | 0.368 | 0.720 | 0.072 | 0.012 | AP 较好，朝向和尺度误差偏高 |
| bicycle | 0.147 | 0.250 | 0.199 | 1.020 | 0.289 | 0.001 | 主要短板，AP 低且朝向误差高 |
| traffic_cone | 0.735 | 0.093 | 0.410 | nan | nan | nan | AP 较好，尺度误差偏高 |
| barrier | 0.000 | 1.000 | 1.000 | 1.000 | nan | nan | mini val 无有效 GT，官方 AP 计 0 |

### 与 FSF baseline 对比

| metric | FSF baseline | FSF_Occ opt1 | 变化 |
| --- | ---: | ---: | ---: |
| mAP | 0.4970 | 0.5089 | +0.0119 |
| NDS | 0.5207 | 0.5353 | +0.0146 |
| present-class mAP | 0.7099 | 0.7270 | +0.0171 |

| class | FSF baseline AP | FSF_Occ opt1 AP | 变化 |
| --- | ---: | ---: | ---: |
| car | 0.867 | 0.865 | -0.002 |
| truck | 0.765 | 0.755 | -0.010 |
| bus | 0.960 | 0.962 | +0.002 |
| trailer | 0.000 | 0.000 | +0.000 |
| construction_vehicle | 0.000 | 0.000 | +0.000 |
| bicycle | 0.102 | 0.147 | +0.045 |
| motorcycle | 0.700 | 0.721 | +0.021 |
| pedestrian | 0.913 | 0.904 | -0.009 |
| traffic_cone | 0.662 | 0.735 | +0.073 |
| barrier | 0.000 | 0.000 | +0.000 |

### 诊断

- `FSF_Occ opt1` 相比原始 FSF 有稳定收益：NDS +0.0146，mAP +0.0119，present-class mAP +0.0171。
- 改进主要来自 `bicycle/motorcycle/traffic_cone`，说明 occupancy-guided completion 对小目标和稀疏目标有正向作用。
- `bicycle` 仍是当前最明显短板：AP 仅 0.147，AOE 达 1.020。该类点数少、局部形状弱，容易被 occupancy refine 阈值过滤到只剩极少点，导致实例几何和朝向线索不足。
- `trailer/construction_vehicle/barrier` 的 0 AP 不能直接解释为模型结构失败，因为 mini val 中这三类没有有效 GT。后续论文或报告中应同时给出 official 10-class mAP 和 present-class mAP，避免误判。
- `frustum_occ_loss_occ_coarse/refine` 在训练后期约 0.008-0.010，过滤头已经较快收敛；继续加大 occ loss 的收益可能有限，优先改过滤后的实例信息保留更合理。

## 改进 2: 稀疏实例 top-K 保点

### 修改动机

原实现中 `ensure_minimum_points_per_instance` 只保证每个实例至少保留 1 个点。对于 `bicycle/motorcycle/traffic_cone` 等稀疏目标，1 个点不足以支撑 completion head 的 size/visibility/center descriptor，也会削弱后续 frustum query 的几何表达。

### 已实现修改

- `ensure_minimum_points_per_instance` 新增 `min_points` 参数，按每个实例的 refine occupancy score 补齐 top-K 高分点。
- `FrustumOccFilter` 新增 `min_points_per_instance` 配置项，默认值为 1，保持旧行为兼容。
- `FSF_Occ` 从 `frustum_occ_filter_cfg` 读取 `min_points_per_instance` 并传入过滤器。
- mini 配置中设置 `min_points_per_instance=2`，让每个 frustum 实例至少保留 2 个高分点。
- 新增测试覆盖 top-K 保点行为和配置传递。

### 预期收益

- 降低 occupancy refine 阈值对少点目标的误伤。
- 提升 completion descriptor 中 `mean/max occupancy`、size residual、visibility 的稳定性。
- 优先改善 `bicycle` 的 AP 和 AOE，同时观察 `motorcycle/traffic_cone` 是否继续受益。
- 风险是背景点略增，可能影响高频类 precision；因此本轮只从 `K=2` 起步，不直接放大到更高保点数。

### 验证

已先写失败测试，再实现代码。单测验证命令：

```bash
wsl --cd /home/ddd/pc/FullySparseFusion bash -lc "source /home/ddd/miniconda3/etc/profile.d/conda.sh && conda activate FSF && pytest tests/test_frustum_occ_filter.py::FrustumOccFilterTests::test_instance_fallback_can_keep_topk_points_per_instance tests/test_frustum_occ_filter.py::FrustumOccFilterTests::test_filter_uses_configured_min_points_per_instance -q"
```

结果：`2 passed`。存在第三方库 deprecation warning，不影响测试结果。

### opt2 训练命令

```bash
wsl --cd /home/ddd/pc/FullySparseFusion bash -lc "source /home/ddd/miniconda3/etc/profile.d/conda.sh && conda activate FSF && PYTHONPATH=.:$PYTHONPATH python tools/train.py projects/configs/nuScenes/FSF_Occ_nuScenes_mini_config.py --work-dir work_dirs/nuScenes/FSF_Occ_nuScenes_mini_config_opt2 --cfg-options evaluation.jsonfile_prefix=work_dirs/nuScenes/FSF_Occ_nuScenes_mini_config_opt2/eval/results"
```

## 实验 2: FSF_Occ opt2

结果来源：`work_dirs/nuScenes/FSF_Occ_nuScenes_mini_config_opt2/20260506_131217.log.json`。本次训练从完整的 `epoch_3.pth` 恢复，跳过中间 checkpoint 频繁写入后完成到 epoch 6。最佳 checkpoint 为 `best_pts_bbox_NuScenes/NDS_epoch_6.pth`，最佳 NDS 出现在 epoch 6。

### 整体指标

| mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5106 | 0.5358 | 0.4239 | 0.4650 | 0.6156 | 0.3942 | 0.2962 |

### 与 opt1 对比

| metric | opt1 | opt2 | 变化 | 结论 |
| --- | ---: | ---: | ---: | --- |
| mAP | 0.5089 | 0.5106 | +0.0017 | 极小提升 |
| NDS | 0.5353 | 0.5358 | +0.0005 | 基本持平 |
| present-class mAP | 0.7270 | 0.7296 | +0.0026 | 有效类别轻微提升 |
| mATE | 0.4253 | 0.4239 | -0.0014 | 定位基本持平 |
| mASE | 0.4709 | 0.4650 | -0.0059 | 尺度略有改善 |
| mAOE | 0.5844 | 0.6156 | +0.0312 | 朝向明显变差 |
| mAVE | 0.4220 | 0.3942 | -0.0278 | 速度明显改善 |
| mAAE | 0.2893 | 0.2962 | +0.0069 | 属性略变差 |

### 分类别 AP 对比

| class | opt1 AP | opt2 AP | 变化 | 观察 |
| --- | ---: | ---: | ---: | --- |
| car | 0.865 | 0.870 | +0.005 | 高频类小幅提升 |
| truck | 0.755 | 0.792 | +0.037 | 受益最明显，定位和速度也改善 |
| bus | 0.962 | 0.941 | -0.021 | AP 回退，但尺度和朝向变好 |
| trailer | 0.000 | 0.000 | +0.000 | mini val 无有效 GT |
| construction_vehicle | 0.000 | 0.000 | +0.000 | mini val 无有效 GT |
| bicycle | 0.147 | 0.148 | +0.001 | AP 几乎不变，AOE 从 1.020 恶化到 1.270 |
| motorcycle | 0.721 | 0.696 | -0.025 | AP 回退，朝向基本持平 |
| pedestrian | 0.904 | 0.904 | +0.000 | AP 持平，但 AOE/AAE 变差 |
| traffic_cone | 0.735 | 0.756 | +0.021 | 小目标 AP 和 ATE 改善，ASE 仍偏高 |
| barrier | 0.000 | 0.000 | +0.000 | mini val 无有效 GT |

### opt2 诊断

- `min_points_per_instance=2` 的收益主要体现在 `truck`、`traffic_cone` 和整体速度误差，说明多保留一个高分点确实能让部分实例几何和运动估计更稳定。
- `bicycle` 没有得到预期改善：AP 仅从 0.147 到 0.148，但 AOE 从 1.020 恶化到 1.270。说明问题不只是点数不足，而是保留下来的点缺少可靠方向约束，额外点可能引入了朝向噪声。
- `motorcycle` AP 回退 0.025，提示统一 top-K 对所有类别不够稳。稀疏实例保点应按类别或实例可见性自适应，而不是全局固定 K。
- `mASE` 和 `mAVE` 改善，但 `mAOE`、`mAAE` 变差，说明 completion descriptor 更偏向补充尺度/速度线索，尚未提供明确 yaw cue。

## 改进 3: 类别感知 top-K + BEV 方向 cue

### 修改动机

opt2 的统一 `min_points_per_instance=2` 只带来极小整体收益：mAP +0.0017，NDS +0.0005。收益主要来自 `truck/traffic_cone`，但 `motorcycle` AP 回退，`bicycle` AOE 从 1.020 恶化到 1.270。这个现象说明统一多保点会给部分实例带来额外朝向噪声，当前瓶颈已经从“是否保住点”转向“保住的点能否提供稳定 yaw 线索”。

### 已实现修改

- `ensure_minimum_points_per_instance` 支持按类别覆写保点数：全局默认 `K=1`，小目标类别可单独设置 `K=2`。
- mini 配置设置 `class_min_points_per_instance={5: 2, 6: 2, 8: 2}`，对应 `bicycle/motorcycle/traffic_cone`；其它类别回退到 `K=1`，避免统一 K=2 影响中高频类 precision。
- `FSF_Occ` 新增 SIR 点到 2D mask 类别的映射，将 `mask_anno[..., 5]` 中的类别 id 传入 `FrustumOccFilter`。
- `FrustumOccFilter` 新增 BEV 主方向 cue：对每个实例的有效点计算 BEV 协方差主轴，并以 `cos(theta), sin(theta)` 追加到 completion descriptor。
- completion descriptor 维度从 8 扩到 10；`frustum_obj_head.in_channel` 和 `mlp_cfg.lidar_img_input_dim` 通过 `_completion_descriptor_dim` 同步更新。

### opt3 训练命令

```bash
wsl --cd /home/ddd/pc/FullySparseFusion bash -lc "source /home/ddd/miniconda3/etc/profile.d/conda.sh && conda activate FSF && PYTHONPATH=.:$PYTHONPATH python tools/train.py projects/configs/nuScenes/FSF_Occ_nuScenes_mini_config.py --work-dir work_dirs/nuScenes/FSF_Occ_nuScenes_mini_config_opt3 --cfg-options evaluation.jsonfile_prefix=work_dirs/nuScenes/FSF_Occ_nuScenes_mini_config_opt3/eval/results"
```

如果 WSL 训练在保存 checkpoint 时中断，建议继续使用上一轮已验证的恢复方式，从完整 checkpoint 恢复并减少中间保存风险：

```bash
wsl --cd /home/ddd/pc/FullySparseFusion bash -lc "source /home/ddd/miniconda3/etc/profile.d/conda.sh && conda activate FSF && PYTHONPATH=.:$PYTHONPATH python tools/train.py projects/configs/nuScenes/FSF_Occ_nuScenes_mini_config.py --work-dir work_dirs/nuScenes/FSF_Occ_nuScenes_mini_config_opt3 --resume-from work_dirs/nuScenes/FSF_Occ_nuScenes_mini_config_opt3/epoch_3.pth --cfg-options evaluation.jsonfile_prefix=work_dirs/nuScenes/FSF_Occ_nuScenes_mini_config_opt3/eval/results"
```

### 验证

已先补充失败测试，再实现代码。验证命令：

```bash
wsl --cd /home/ddd/pc/FullySparseFusion bash -lc "source /home/ddd/miniconda3/etc/profile.d/conda.sh && conda activate FSF && pytest tests/test_frustum_occ_filter.py::FrustumOccFilterTests::test_instance_fallback_uses_class_specific_min_points tests/test_frustum_occ_filter.py::FrustumOccFilterTests::test_bev_orientation_cues_follow_instance_major_axis tests/test_fsf_occ_completion.py -q"
```

结果：`4 passed`。存在第三方库 deprecation warning，不影响测试结果。

## 下一步

- 运行 opt3，在 `nuscenes_mini` 上重点比较 `bicycle AOE`、`motorcycle AP/AOE`、`traffic_cone ASE`、`truck AP` 和 `present-class mAP`。
- 如果 `bicycle AOE` 仍高，下一轮不再继续增大 top-K，优先考虑将方向 cue 改为有符号长轴/短轴置信度，或对 `bicycle/motorcycle` 增加 yaw-aware descriptor loss。
- 如果 `truck` 的 opt2 收益消失，说明 class-aware K 过于保守，可只对 `truck` 追加 `K=2` 做小范围消融，而不是恢复全类别 K=2。
