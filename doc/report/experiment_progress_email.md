# FSF_Occ nuScenes Mini 实验进展汇报

**主题：** FSF_Occ nuScenes Mini 实验改进与结果总结

**日期：** 2026-05-10

---

各位好，

以下是 FSF_Occ 在 nuScenes mini 数据集上的实验改进与结果总结，供参考。

---

## 1. 实验背景

- **数据集：** nuScenes mini (`data/nuscenes_mini/`)
- **评估指标：** mAP、NDS 及 TP error 指标 (mATE/mASE/mAOE/mAVE/mAAE)
- **注意事项：** mini val 中 `trailer/construction_vehicle/barrier` 无有效 GT，官方 10 类 mAP 会被这三个 0 AP 类别拉低。因此同时记录 **present-class mAP**（仅对有有效 val GT 的 7 个类别计算）作为辅助判断。

---

## 2. 改进路线与实验结果

### Baseline

| mAP | NDS | present-class mAP |
| ---: | ---: | ---: |
| 0.4970 | 0.5207 | 0.7099 |

### 改进 1 (opt1): Coarse-to-Fine Occupancy + 实例级 Completion Descriptor + 辅助损失

**核心改动：**
- 引入 coarse-to-fine occupancy 机制
- 实例级 completion descriptor（含 size/visibility 辅助损失）

**结果：**

| mAP | NDS | present-class mAP |
| ---: | ---: | ---: |
| 0.5089 | 0.5353 | 0.7270 |

**vs Baseline：**

| 指标 | Baseline | opt1 | 变化 |
| --- | ---: | ---: | ---: |
| mAP | 0.4970 | 0.5089 | **+0.0119** |
| NDS | 0.5207 | 0.5353 | **+0.0146** |
| present-class mAP | 0.7099 | 0.7270 | **+0.0171** |

**关键观察：**
- 改进主要来自 `bicycle (+0.045)`、`motorcycle (+0.021)`、`traffic_cone (+0.073)`，说明 occupancy-guided completion 对小目标和稀疏目标有正向作用
- `bicycle` 仍是短板：AP 仅 0.147，AOE 达 1.020

---

### 改进 2 (opt2): 稀疏实例 Top-K 保点 (min_points_per_instance=2)

**核心改动：**
- 每个实例至少保留 2 个高分点，降低 occupancy refine 阈值对少点目标的误伤

**结果：**

| mAP | NDS | present-class mAP |
| ---: | ---: | ---: |
| 0.5106 | 0.5358 | 0.7296 |

**vs opt1：**

| 指标 | opt1 | opt2 | 变化 | 结论 |
| --- | ---: | ---: | ---: | --- |
| mAP | 0.5089 | 0.5106 | +0.0017 | 极小提升 |
| NDS | 0.5353 | 0.5358 | +0.0005 | 基本持平 |
| present-class mAP | 0.7270 | 0.7296 | +0.0026 | 轻微提升 |
| mASE | 0.4709 | 0.4650 | -0.0059 | 尺度略有改善 |
| mAOE | 0.5844 | 0.6156 | +0.0312 | **朝向明显变差** |
| mAVE | 0.4220 | 0.3942 | -0.0278 | **速度明显改善** |

**关键观察：**
- 收益主要来自 `truck (+0.037 AP)`、`traffic_cone (+0.021 AP)` 和速度误差改善
- `bicycle` AP 几乎不变 (0.147→0.148)，AOE 从 1.020 **恶化到 1.270**
- `motorcycle` AP 回退 0.025
- 统一 K=2 会给部分实例引入额外朝向噪声，瓶颈已从"是否保住点"转向"保住的点能否提供稳定 yaw 线索"

---

### 改进 3 (opt3): 类别感知 Top-K + BEV 方向 Cue（待验证）

**核心改动：**
- 按类别设置保点数：`bicycle/motorcycle/traffic_cone` 设 K=2，其它类别回退 K=1，避免统一 K=2 影响高频类 precision
- 新增 BEV 主方向 cue：对每个实例计算 BEV 协方差主轴，以 `(cos θ, sin θ)` 追加到 completion descriptor
- Completion descriptor 维度从 8 扩到 10，同步更新 `frustum_obj_head.in_channel` 和 `mlp_cfg.lidar_img_input_dim`

**预期收益：**
- 类别感知 K 避免中高频类 precision 下降
- BEV 方向 cue 为 completion descriptor 提供显式 yaw 线索，优先改善 `bicycle` AOE
- 风险：BEV 主轴对极稀疏实例可能不稳定

**状态：** 代码已实现，单测通过（4 passed），待训练验证

---

## 3. 三轮实验结果汇总

| 版本 | 主要改动 | mAP | NDS | present-class mAP | mAOE | mAVE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 原始 FSF | 0.4970 | 0.5207 | 0.7099 | — | — |
| opt1 | Occ + Completion Descriptor | 0.5089 | 0.5353 | 0.7270 | 0.5844 | 0.4220 |
| opt2 | Top-K 保点 K=2 | 0.5106 | 0.5358 | 0.7296 | 0.6156 | 0.3942 |
| opt3 | 类别感知 K + BEV 方向 cue | 待验证 | 待验证 | 待验证 | 待验证 | 待验证 |

**累计改进 (Baseline → opt2)：** mAP +0.0136，NDS +0.0151，present-class mAP +0.0197

---

## 4. 当前瓶颈与下一步

- **核心瓶颈：** `bicycle` AOE 持续偏高（1.020→1.270），稀疏实例的朝向估计缺乏可靠约束
- **opt3 验证重点：** `bicycle AOE`、`motorcycle AP/AOE`、`traffic_cone ASE`、`truck AP`、`present-class mAP`
- **如果 opt3 bicycle AOE 仍高：** 不再增大 top-K，优先考虑将方向 cue 改为有符号长轴/短轴置信度，或对 `bicycle/motorcycle` 增加 yaw-aware descriptor loss

---

如有问题或建议，欢迎讨论。
