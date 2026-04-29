# Research 1 Lab: FSF_HierMamba nuScenes Mini 实验记录

本文档用于记录 `FSF_HierMamba` 在 nuScenes mini 上的每次改进点、训练/评估结果和下一步优化依据。

## 记录说明

- 数据集：nuScenes mini。
- 主要指标：mAP、NDS，以及 nuScenes TP error 指标 `mATE/mASE/mAOE/mAVE/mAAE`。
- mini split 的类别分布不完整。当前 val 统计中存在有效 GT 的类别主要是 `car/truck/bus/pedestrian/motorcycle/bicycle/traffic_cone`，`trailer/construction_vehicle/barrier` 在该 split 中没有有效 GT，因此官方 mAP 会被这些 0 AP 类别显著拉低。
- 为了辅助判断真实改进，本文同时记录 present-class mAP，即只对有有效 GT 的 7 个类别计算的 AP 均值。

## 实验总览

| 版本 | 配置/结果目录 | 主要改进点 | mAP | NDS | present-class mAP | 状态 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| baseline | `FSF_HierMamba_nuScenes_mini_config` | 原始 HierMamba 融合配置 | 0.5112 | 0.5424 | 0.7303 | 已完成 |
| opt1 | `FSF_HierMamba_nuScenes_mini_config_opt` | 类别感知 token 保留、扩展 reliability descriptor、小目标残差增强 | 0.5284 | 0.5524 | 0.7548 | 已完成 |
| opt2 | `FSF_HierMamba_nuScenes_mini_config_opt2` | mini-specific 类别配额收窄、类别配额缺口修正 | 待训练 | 待训练 | 待训练 | 待训练 |

## 实验 0: baseline

结果来源：`work_dirs/nuScenes/FSF_HierMamba_nuScenes_mini_config/eval/results/metrics_summary.json`，评估时间 `2026-04-28 19:59`。

### 整体指标

| mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5112 | 0.5424 | 0.4259 | 0.4686 | 0.5493 | 0.3981 | 0.2902 |

### 分类 AP

| class | AP | 主要观察 |
| --- | ---: | --- |
| car | 0.867 | 高频类表现稳定，定位和朝向误差较低 |
| truck | 0.800 | 表现较好 |
| bus | 0.981 | AP 很高，但 mini 中样本量小，波动风险高 |
| trailer | 0.000 | mini val 无有效 GT，官方均值中计 0 |
| construction_vehicle | 0.000 | mini val 无有效 GT，官方均值中计 0 |
| pedestrian | 0.906 | 表现较强 |
| motorcycle | 0.674 | AP 尚可，但朝向误差偏高 |
| bicycle | 0.217 | 主要短板，AP 低且朝向误差高 |
| traffic_cone | 0.668 | AP 可提升，scale error 偏高 |
| barrier | 0.000 | mini val 无有效 GT，官方均值中计 0 |

### baseline 诊断

- 官方 mAP 被 `trailer/construction_vehicle/barrier` 三个无有效 GT 类别显著拉低，present-class mAP 为 0.7303。
- 真正需要优先优化的是小目标和稀有有效类：`bicycle`、`motorcycle`、`traffic_cone`。
- `bicycle` 的 AP 只有 0.217，AOE 达 0.791，说明 token 保留和朝向建模不足。
- `traffic_cone` AP 为 0.668，ASE 为 0.397，说明小目标尺度估计仍有优化空间。
- 虽然 `trailer/construction_vehicle/barrier` 在 mini val 中无有效 GT，但模型仍会预测这些类别，容易引入 no-GT 类别的误检和训练/评估噪声。

## 改进 1: 类别感知小目标增强

### 修改点

- 在 `ForegroundTokenSelector` 中增加 `class_groups` 和 `class_group_min_tokens`，让小目标/长尾类别在全局 top-k token 选择之前获得最低 token 保留配额。
- `HierarchicalHMambaInteraction` 向 token selector 传递类别组配置。
- reliability descriptor 从 6 维扩展到 9 维，增加距离置信度、2D aspect cue、投影一致性 cue。
- 增加小目标残差分支，输出 `small_object_mask`、`small_object_pose_residual`、`small_object_scale_residual`、`small_object_temperature`，用于对小目标 token 做轻量 gated feature delta。
- nuScenes mini/full 初始配置使用：
  - `class_groups=[[2,4],[5,6,8,9]]`
  - `class_group_min_tokens=[1,2]`
  - `small_object_class_indices=[2,4,5,6,8,9]`
- Argoverse2 配置按 AV2 类别顺序单独映射，避免直接复用 nuScenes 类别索引。

### 验证

```powershell
C:\Users\1\miniconda3\envs\FSF\python.exe -m unittest tests.test_hierarchical_hmamba tests.test_fsf_hiermamba_detector
```

结果：9 个单元测试通过。源文件编译检查通过。全量 unittest discover 因无关 `frustum_occ_filter` 测试缺少 `mmcv` 依赖而未作为本轮通过标准。

## 实验 1: opt1

结果来源：`work_dirs/nuScenes/FSF_HierMamba_nuScenes_mini_config_opt/eval/results/metrics_summary.json`，评估时间 `2026-04-29 02:11`。

### 整体指标对比

| metric | baseline | opt1 | 变化 |
| --- | ---: | ---: | ---: |
| mAP | 0.5112 | 0.5284 | +0.0171 |
| NDS | 0.5424 | 0.5524 | +0.0100 |
| mATE | 0.4259 | 0.4165 | -0.0094 |
| mASE | 0.4686 | 0.4637 | -0.0049 |
| mAOE | 0.5493 | 0.5372 | -0.0121 |
| mAVE | 0.3981 | 0.3978 | -0.0003 |
| mAAE | 0.2902 | 0.3027 | +0.0125 |
| present-class mAP | 0.7303 | 0.7548 | +0.0245 |

### 分类 AP 对比

| class | baseline AP | opt1 AP | 变化 |
| --- | ---: | ---: | ---: |
| car | 0.867 | 0.873 | +0.006 |
| truck | 0.800 | 0.784 | -0.017 |
| bus | 0.981 | 0.945 | -0.035 |
| trailer | 0.000 | 0.000 | +0.000 |
| construction_vehicle | 0.000 | 0.000 | +0.000 |
| pedestrian | 0.906 | 0.904 | -0.002 |
| motorcycle | 0.674 | 0.679 | +0.005 |
| bicycle | 0.217 | 0.349 | +0.133 |
| traffic_cone | 0.668 | 0.749 | +0.081 |
| barrier | 0.000 | 0.000 | +0.000 |

### opt1 结论

- opt1 对目标优化方向有效：`bicycle` AP 提升 0.133，`traffic_cone` AP 提升 0.081，present-class mAP 提升 0.0245。
- 定位、尺度、朝向整体误差均下降，NDS 提升 0.0100。
- `bus` 和 `truck` 出现回退，且 `mAAE` 变差。mini 中 bus/truck 样本量较小，波动会被放大，但也说明类别配额和小目标残差分支可能对非目标类产生了干扰。
- opt1 的 mini 配置把 `trailer/construction_vehicle/barrier` 也纳入 token quota 或 small-object residual 相关集合，但这些类在 mini val 没有有效 GT，容易浪费 token 配额并放大 no-GT 类别误检。

## 改进 2: mini-specific 配额收窄与配额缺口修正

### 修改点

- mini 配置只保留当前 split 有有效 GT 且确实需要增强的类别：
  - `class_groups=[[5,6,8]]`
  - `class_group_min_tokens=[2]`
  - `small_object_class_indices=[5,6,8]`
- full nuScenes 配置保持更宽的长尾/小目标设置，不把 mini 的数据分布假设强行带到 full 训练。
- 修正 `_apply_class_group_quotas`：
  - 先统计 `min_per_modality` 已保留 token 中属于目标 class group 的数量。
  - 只补齐 quota 缺口，避免已经满足配额时继续额外抢占 token。
  - quota 填充时只从预测类别属于该 group 的候选 token 中选择，避免用 group score 高但类别不匹配的 token 填充小目标配额。
- 新增测试覆盖：
  - `test_class_group_quota_counts_tokens_kept_by_modality_floor`
  - `tests/test_hiermamba_configs.py`，确认 mini 配置只面向 `{5,6,8}`。

### 验证

```powershell
C:\Users\1\miniconda3\envs\FSF\python.exe -m unittest tests.test_hierarchical_hmamba tests.test_hiermamba_configs tests.test_fsf_hiermamba_detector
```

结果：11 个单元测试通过。相关源码编译检查通过。`git diff --check` 通过，仅存在 CRLF warning。

### 预期收益

- 减少 `trailer/construction_vehicle/barrier` 这类 mini no-GT 类别对 token 配额和残差分支的干扰。
- 保留 opt1 对 `bicycle/motorcycle/traffic_cone` 的收益，同时尝试恢复 `truck/bus` 的 AP。
- 降低 no-GT 类别预测数，改善 mini split 下的误检噪声。

### 重新训练命令

```powershell
conda activate FSF
$env:PYTHONPATH="$(Get-Location);$env:PYTHONPATH"
python tools/train.py projects/configs/nuScenes/FSF_HierMamba_nuScenes_mini_config.py --work-dir work_dirs/nuScenes/FSF_HierMamba_nuScenes_mini_config_opt2 --cfg-options evaluation.jsonfile_prefix=work_dirs/nuScenes/FSF_HierMamba_nuScenes_mini_config_opt2/eval/results
```

### 实验 2 结果

当前状态：待训练。

| metric | opt1 | opt2 | 变化 |
| --- | ---: | ---: | ---: |
| mAP | 0.5284 | 待训练 | 待训练 |
| NDS | 0.5524 | 待训练 | 待训练 |
| present-class mAP | 0.7548 | 待训练 | 待训练 |
| bicycle AP | 0.349 | 待训练 | 待训练 |
| traffic_cone AP | 0.749 | 待训练 | 待训练 |
| truck AP | 0.784 | 待训练 | 待训练 |
| bus AP | 0.945 | 待训练 | 待训练 |

## 后续优化优先级

1. 先训练 opt2，确认 mini-specific 配额收窄是否能在保留 `bicycle/traffic_cone` 收益的同时恢复 `truck/bus`。
2. 若 opt2 的 `bicycle` 仍明显弱于其他有效类，优先检查 bicycle 的朝向分支和正样本分配，而不是继续增加全局 token 数。
3. 若 `traffic_cone` ASE 仍高，考虑对小目标尺寸残差分支增加更直接的监督或类别条件化尺度先验。
4. 若 no-GT 类别预测数仍高，在 mini 实验中可单独加入类别先验校准或后处理阈值分析；最终结论仍应以 full nuScenes val 为准。

