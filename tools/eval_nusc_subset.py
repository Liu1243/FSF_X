import argparse
import json
import os
from collections import OrderedDict

import mmcv
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import (
    DetectionBox,
    DetectionMetricDataList,
    DetectionMetrics,
)
from nuscenes.eval.detection.evaluate import TP_METRICS, NuScenesEval
from nuscenes.eval.common.loaders import (
    add_center_dist,
    filter_eval_boxes,
    load_gt,
    load_prediction,
)


ERR_NAME_MAPPING = {
    "trans_err": "mATE",
    "scale_err": "mASE",
    "orient_err": "mAOE",
    "vel_err": "mAVE",
    "attr_err": "mAAE",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate nuScenes predictions on a token subset.")
    parser.add_argument("--result", required=True, help="Path to results_nusc.json.")
    parser.add_argument("--dataroot", required=True, help="nuScenes dataroot.")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--eval-set", default="train", choices=["train", "val", "mini_train", "mini_val"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-version", default="detection_cvpr_2019")
    parser.add_argument("--summary-out", required=True)
    return parser.parse_args()


def subset_eval_boxes(eval_boxes, sample_tokens):
    subset = EvalBoxes()
    for token in sample_tokens:
        subset.add_boxes(token, eval_boxes[token])
    return subset


def evaluate_subset(nusc_eval):
    metrics_data = DetectionMetricDataList()
    for class_name in nusc_eval.cfg.class_names:
        for dist_th in nusc_eval.cfg.dist_ths:
            metric_data = accumulate(
                nusc_eval.gt_boxes,
                nusc_eval.pred_boxes,
                class_name,
                nusc_eval.cfg.dist_fcn_callable,
                dist_th,
            )
            metrics_data.set(class_name, dist_th, metric_data)

    metrics = DetectionMetrics(nusc_eval.cfg)
    for class_name in nusc_eval.cfg.class_names:
        for dist_th in nusc_eval.cfg.dist_ths:
            metric_data = metrics_data[(class_name, dist_th)]
            metrics.add_label_ap(
                class_name,
                dist_th,
                calc_ap(metric_data, nusc_eval.cfg.min_recall, nusc_eval.cfg.min_precision),
            )

        for metric_name in TP_METRICS:
            metric_data = metrics_data[(class_name, nusc_eval.cfg.dist_th_tp)]
            if class_name == "traffic_cone" and metric_name in ["attr_err", "vel_err", "orient_err"]:
                tp = float("nan")
            elif class_name == "barrier" and metric_name in ["attr_err", "vel_err"]:
                tp = float("nan")
            else:
                tp = calc_tp(metric_data, nusc_eval.cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)

    return metrics, metrics_data


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.output_dir)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    cfg = config_factory(args.eval_version)

    pred_boxes, meta = load_prediction(args.result, cfg.max_boxes_per_sample, DetectionBox, verbose=True)
    all_gt_boxes = load_gt(nusc, args.eval_set, DetectionBox, verbose=True)
    sample_tokens = list(pred_boxes.sample_tokens)
    missing_tokens = sorted(set(sample_tokens) - set(all_gt_boxes.sample_tokens))
    if missing_tokens:
        raise RuntimeError(f"{len(missing_tokens)} prediction tokens are not in eval set {args.eval_set}: {missing_tokens[:5]}")

    gt_boxes = subset_eval_boxes(all_gt_boxes, sample_tokens)

    nusc_eval = NuScenesEval.__new__(NuScenesEval)
    nusc_eval.nusc = nusc
    nusc_eval.result_path = args.result
    nusc_eval.eval_set = args.eval_set
    nusc_eval.output_dir = args.output_dir
    nusc_eval.verbose = True
    nusc_eval.cfg = cfg
    nusc_eval.plot_dir = os.path.join(args.output_dir, "plots")
    mmcv.mkdir_or_exist(nusc_eval.plot_dir)
    nusc_eval.pred_boxes = add_center_dist(nusc, pred_boxes)
    nusc_eval.gt_boxes = add_center_dist(nusc, gt_boxes)
    nusc_eval.pred_boxes = filter_eval_boxes(nusc, nusc_eval.pred_boxes, cfg.class_range, verbose=True)
    nusc_eval.gt_boxes = filter_eval_boxes(nusc, nusc_eval.gt_boxes, cfg.class_range, verbose=True)
    nusc_eval.sample_tokens = nusc_eval.gt_boxes.sample_tokens
    nusc_eval.meta = meta

    metrics, metrics_data = evaluate_subset(nusc_eval)
    metrics_summary = metrics.serialize()
    metrics_summary["meta"] = meta.copy()
    metrics_summary["eval_set"] = args.eval_set
    metrics_summary["num_samples"] = len(sample_tokens)

    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
    with open(os.path.join(args.output_dir, "metrics_details.json"), "w") as f:
        json.dump(metrics_data.serialize(), f, indent=2)

    detail = OrderedDict()
    metric_prefix = "pts_bbox_NuScenes"
    for name in cfg.class_names:
        for k, v in metrics_summary["label_aps"][name].items():
            detail[f"{metric_prefix}/{name}_AP_dist_{k}"] = float(f"{v:.4f}")
        for k, v in metrics_summary["label_tp_errors"][name].items():
            detail[f"{metric_prefix}/{name}_{k}"] = float(f"{v:.4f}")
    for k, v in metrics_summary["tp_errors"].items():
        detail[f"{metric_prefix}/{ERR_NAME_MAPPING[k]}"] = float(f"{v:.4f}")
    detail[f"{metric_prefix}/NDS"] = metrics_summary["nd_score"]
    detail[f"{metric_prefix}/mAP"] = metrics_summary["mean_ap"]
    detail["num_samples"] = len(sample_tokens)

    mmcv.dump(detail, args.summary_out)
    print(json.dumps(detail, indent=2))


if __name__ == "__main__":
    main()
