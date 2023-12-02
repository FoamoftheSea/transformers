from collections import Counter
from typing import Dict, Optional, Set

import numpy as np
import torch
from shift_lab.tools.kitti_object_eval_python.eval import get_shift_coco_eval_result
from torchmetrics.detection import MeanAveragePrecision

from transformers import EvalPrediction, MultiformerConfig
from transformers.models.multiformer.image_processing_multiformer import post_process_object_detection
from transformers.models.multiformer.modeling_multiformer import IRMSELoss, MultiformerTask, SiLogLoss


class MultiformerSemanticEvalMetric:
    def __init__(
        self, id2label: Dict[int, str], ignore_class_ids: Optional[Set[int]] = None, reduced_labels: bool = False
    ):
        self.total_area_intersect = Counter()
        self.total_area_union = Counter()
        self.total_label_area = Counter()
        self.ignore_class_ids = ignore_class_ids or set()
        self.reduced_labels = reduced_labels
        self.id2label = id2label

    def update(self, preds: np.ndarray, target: np.ndarray):
        logits_tensor = torch.from_numpy(preds)
        # scale the logits to the size of the label
        logits_tensor = torch.nn.functional.interpolate(
            logits_tensor,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()

        for class_id in self.id2label.keys():
            if class_id in self.ignore_class_ids:
                continue
            if self.reduced_labels:
                label_id = class_id - 1 if class_id != 0 else 255
            else:
                label_id = class_id
            pred_pixels = pred_labels == label_id
            gt_pixels = target == label_id
            class_label = self.id2label[class_id]
            self.total_area_intersect.update({class_label: np.sum(np.bitwise_and(pred_pixels, gt_pixels))})
            self.total_area_union.update({class_label: np.sum(np.bitwise_or(pred_pixels, gt_pixels))})
            self.total_label_area.update({class_label: np.sum(gt_pixels)})

    def compute(self):
        accuracies = {
            f"accuracy_{k}": self.total_area_intersect[k] / self.total_label_area[k] for k in self.total_area_union
        }
        ious = {f"iou_{k}": self.total_area_intersect[k] / self.total_area_union[k] for k in self.total_area_union}
        metrics = {
            "overall_accuracy": sum(self.total_area_intersect.values()) / sum(self.total_label_area.values()),
            "mean_accuracy": np.mean(list(accuracies.values())),
            "mean_iou": np.mean(list(ious.values())),
        }
        metrics.update(accuracies)
        metrics.update(ious)

        return metrics


class MultiformerDet3DEvalMetric:
    def __init__(self, config: MultiformerConfig):
        self.gt_annos = []
        self.dt_annos = []
        self.config = config
        self.num_heading_bins = config.det3d_num_heading_bins
        self.type_mean_size_array = np.zeros((config.num_labels, 3))
        self.type_mean_id_map: Dict[int, int] = {}
        for i, class_id in enumerate(sorted(config.id2label.keys())):
            self.type_mean_size_array[i] = np.array(config.det3d_type_mean_sizes[config.id2label[class_id]])
            self.type_mean_id_map[class_id] = i

    def update(self, preds, target):
        for batch_item in preds:
            boxes3d = batch_item["boxes3d"]
            # Convert heading to radians
            heading_class_scores = boxes3d[:, 3:self.num_heading_bins + 3]
            heading_class_pred = heading_class_scores.argmax(-1)
            hcls_onehot = np.zeros((heading_class_pred.shape[0], self.num_heading_bins))
            hcls_onehot[np.arange(hcls_onehot.shape[0]), heading_class_pred] = 1
            heading_residual_normalized_pred = boxes3d[:, self.num_heading_bins + 3:self.num_heading_bins*2 + 3]
            heading_bin_centers = np.arange(0, 2 * np.pi, 2 * np.pi / self.num_heading_bins)
            heading_pred = heading_residual_normalized_pred * (np.pi / self.num_heading_bins) + heading_bin_centers
            heading_pred = np.sum(heading_pred * hcls_onehot, 1)

            # Get dimensions in meters
            if self.config.det3d_predict_class:
                size_class_scores = boxes3d[:, -4*self.config.num_labels:-3*self.config.num_labels]
                size_class_preds = size_class_scores.argmax(-1)
            else:
                size_class_preds = boxes3d[:, -1].astype(np.int64)
                boxes3d = boxes3d[:, :-1]
            scls_onehot = np.zeros((size_class_preds.shape[0], self.config.num_labels))
            scls_onehot[np.arange(scls_onehot.shape[0]), size_class_preds] = 1
            scls_onehot = scls_onehot.reshape(size_class_preds.shape[0], self.config.num_labels, 1).repeat(3, -1)
            size_residual_normalized_pred = boxes3d[:, -3*self.config.num_labels:]
            size_residuals_pred = size_residual_normalized_pred.reshape(scls_onehot.shape) * self.type_mean_size_array
            size_pred = size_residuals_pred + self.type_mean_size_array
            size_pred = np.sum(size_pred * scls_onehot, 1)

            self.dt_annos.append(
                {
                    "name": [self.config.id2label[class_id] for class_id in size_class_preds],
                    "bbox": batch_item["boxes2d"].detach().cpu().numpy(),
                    "score": batch_item["score"],
                    "location": boxes3d[:, :3],
                    "dimensions": size_pred,
                    "rotation_y": heading_pred,
                }
            )
        for batch_item in target:
            self.gt_annos.append(
                {
                    "name": [self.config.id2label[class_id] for class_id in batch_item["class_labels"]],
                    "bbox": batch_item["boxes2d"].detach().cpu().numpy(),
                    "location": batch_item["boxes3d"][:, :3],
                    "dimensions": batch_item["boxes3d"][:, 3:6],
                    "rotation_y": batch_item["boxes3d"][:, -2],
                }
            )

    def compute(self):
        result, log_dict = get_shift_coco_eval_result(self.gt_annos, self.dt_annos, list(self.config.label2id.keys()))

        return log_dict


class MultiformerDepthEvalMetric:
    def __init__(self, silog_lambda=0.5, log_predictions=True, log_labels=False, mask_value=0.0):
        self.batch_mae = []
        self.batch_mse = []
        self.batch_rmse = []
        self.batch_irmse = []
        self.batch_silog = []
        self.irmse_loss = IRMSELoss(log_predictions=log_predictions, log_labels=log_labels)
        self.silog_loss = SiLogLoss(lambd=silog_lambda, log_predictions=log_predictions, log_labels=log_labels)
        self.log_predictions = log_predictions
        self.log_labels = log_labels
        self.mask_value = mask_value

    def update(self, preds, target):
        valid_pixels = np.where(target != self.mask_value)
        y = np.exp(target[valid_pixels]) if self.log_labels else target[valid_pixels]
        y_hat = np.exp(preds[valid_pixels]) if self.log_predictions else preds[valid_pixels]
        valid_mask = y > 0
        diff = y[valid_mask] - y_hat[valid_mask]
        self.batch_mae.append(np.mean(np.abs(diff)))
        batch_mse = np.mean(np.power(diff, 2))
        self.batch_mse.append(batch_mse)
        self.batch_rmse.append(np.sqrt(batch_mse))
        self.batch_irmse.append(self.irmse_loss(torch.from_numpy(preds), torch.from_numpy(target)))
        self.batch_silog.append(self.silog_loss(torch.from_numpy(preds), torch.from_numpy(target)))

    def compute(self):
        return {
            "mae": np.mean(self.batch_mae),
            "mse": np.mean(self.batch_mse),
            "rmse": np.mean(self.batch_rmse),
            "irmse": np.mean(self.batch_irmse),
            "silog": np.mean(self.batch_silog),
        }


class MultiformerMetric:
    def __init__(
        self,
        config: MultiformerConfig,
        id2label: Optional[Dict[int, str]] = None,
        ignore_class_ids: Optional[Set[int]] = None,
        reduced_labels: bool = False,
        box_score_threshold: float = 0.5,
    ):
        self.config = config
        self.tasks = config.tasks
        self.id2label = id2label
        self.ignore_class_ids = ignore_class_ids
        self.reduced_labels = reduced_labels
        self.box_score_threshold = box_score_threshold
        self.metrics = {}
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics = {}
        if "det2d" in self.tasks:
            self.metrics["det_2d"] = MeanAveragePrecision()
        if "det3d" in self.tasks:
            self.metrics["det_3d"] = MultiformerDet3DEvalMetric(self.config)
        if "semseg" in self.tasks:
            self.metrics["semseg"] = MultiformerSemanticEvalMetric(
                id2label=self.id2label, ignore_class_ids=self.ignore_class_ids, reduced_labels=self.reduced_labels
            )
        if "depth" in self.tasks:
            self.metrics["depth"] = MultiformerDepthEvalMetric()

    def convert_eval_pred(self, eval_pred, task):
        if task == MultiformerTask.DET_2D:
            target_sizes = [eval_pred.inputs[i, 0].shape for i in range(eval_pred.inputs.shape[0])]
            preds = post_process_object_detection(
                eval_pred.predictions, threshold=self.box_score_threshold, target_sizes=target_sizes
            )

            target = [
                {
                    "boxes": post_process_object_detection(
                        eval_pred.label_ids["labels"][i]["boxes"][None, ...],
                        threshold=self.box_score_threshold,
                        target_sizes=[target_sizes[i]],
                        boxes_only=True,
                    )[0]["boxes"],
                    "labels": torch.LongTensor(eval_pred.label_ids["labels"][i]["class_labels"]),
                    # "masks": None,
                    # "iscrowd": None,
                    # "area": None,
                }
                for i in range(len(eval_pred.label_ids["labels"]))
            ]

        elif task == MultiformerTask.SEMSEG:
            preds = eval_pred.predictions["logits_semantic"]
            target = eval_pred.label_ids["labels_semantic"]
        elif task == MultiformerTask.DET_3D:
            target_sizes = [eval_pred.inputs[i, 0].shape for i in range(eval_pred.inputs.shape[0])]
            preds_3d = eval_pred.predictions["pred_boxes_3d"]
            if not self.config.det3d_predict_class:
                preds_3d = np.concatenate([preds_3d, eval_pred.predictions["logits"].argmax(-1)[..., None]], axis=-1)
            preds_2d = post_process_object_detection(
                eval_pred.predictions, threshold=0.0, target_sizes=target_sizes, top_k=300
            )
            scores = eval_pred.predictions["logits"].max(-1)
            preds = []
            for pred_3d, pred_2d, score in zip(preds_3d, preds_2d, scores):
                preds.append({"boxes3d": pred_3d, "boxes2d": pred_2d["boxes"], "score": score})
            target = eval_pred.label_ids["labels_3d"]
            for i, t in enumerate(target):
                t["boxes2d"] = post_process_object_detection(
                    eval_pred.label_ids["labels"][i]["boxes"][None, ...],
                    threshold=self.box_score_threshold,
                    target_sizes=[target_sizes[i]],
                    boxes_only=True,
                )[0]["boxes"]
        else:
            preds = eval_pred.predictions["pred_depth"]
            target = eval_pred.label_ids["labels_depth"]

        return preds, target

    def update(self, task: MultiformerTask, eval_pred: EvalPrediction):
        preds, target = self.convert_eval_pred(eval_pred, task)
        self.metrics[task].update(
            preds=preds,
            target=target,
        )

    def compute(self):
        output = {task: metric.compute() for task, metric in self.metrics.items()}
        if output.get("det_2d", None) is not None:
            output["det_2d"]["classes"] = output["det_2d"].pop("classes").tolist()
        self.reset_metrics()
        return output
