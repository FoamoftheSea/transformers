from collections import Counter
from typing import Optional, Dict, Set, List

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

from transformers import EvalPrediction
from transformers.models.multiformer.image_processing_multiformer import post_process_object_detection
from transformers.models.multiformer.modeling_multiformer import MultiformerTask, IRMSELoss, SiLogLoss


class MultiformerSemanticEvalMetric:
    def __init__(
            self,
            id2label: Dict[int, str],
            ignore_class_ids: Optional[Set[int]] = None,
            reduced_labels: bool = False
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
        accuracies = {f"accuracy_{k}": self.total_area_intersect[k] / self.total_label_area[k] for k in self.total_area_union}
        ious = {f"iou_{k}": self.total_area_intersect[k] / self.total_area_union[k] for k in self.total_area_union}
        metrics = {
            "overall_accuracy": sum(self.total_area_intersect.values()) / sum(self.total_label_area.values()),
            "mean_accuracy": np.mean(list(accuracies.values())),
            "mean_iou": np.mean(list(ious.values())),
        }
        metrics.update(accuracies)
        metrics.update(ious)

        return metrics


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
        id2label: Optional[Dict[int, str]] = None,
        ignore_class_ids: Optional[Set[int]] = None,
        reduced_labels: bool = False,
        box_score_threshold: float = 0.35,
    ):
        self.id2label = id2label
        self.ignore_class_ids = ignore_class_ids
        self.reduced_labels = reduced_labels
        self.box_score_threshold = box_score_threshold
        self.metrics = {}
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics = {
            "det_2d": MeanAveragePrecision(),
            "semseg": MultiformerSemanticEvalMetric(
                id2label=self.id2label,
                ignore_class_ids=self.ignore_class_ids,
                reduced_labels=self.reduced_labels
            ),
            "depth": MultiformerDepthEvalMetric(),
        }

    def convert_eval_pred(self, eval_pred, task):
        if task == MultiformerTask.DET_2D:
            target_sizes = [eval_pred.inputs[i, 0].shape for i in range(eval_pred.inputs.shape[0])]
            preds = post_process_object_detection(
                eval_pred.predictions,
                threshold=self.box_score_threshold,
                target_sizes=target_sizes
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
