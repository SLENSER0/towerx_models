import argparse
import torch
import numpy as np
import supervision as sv
import albumentations as A

from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


class MAPEvaluator:
    """
    Evaluator for Mean Average Precision (mAP) using TorchMetrics.
    """

    def __init__(self, image_processor, threshold=0.00, id2label=None):
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label

    def collect_image_sizes(self, targets):
        """
        Collect image sizes from targets.

        Args:
            targets (list): List of target dictionaries.

        Returns:
            list: List of image size tensors.
        """
        image_sizes = []
        for batch in targets:
            batch_image_sizes = torch.tensor([x["size"] for x in batch])
            image_sizes.append(batch_image_sizes)
        return image_sizes

    def collect_targets(self, targets, image_sizes):
        """
        Process and collect targets for evaluation.

        Args:
            targets (list): List of target dictionaries.
            image_sizes (list): List of image size tensors.

        Returns:
            list: List of processed target dictionaries.
        """
        post_processed_targets = []
        for target_batch, image_size_batch in zip(targets, image_sizes):
            for target, (height, width) in zip(target_batch, image_size_batch):
                boxes = target["boxes"]
                boxes = sv.xcycwh_to_xyxy(boxes)
                boxes = boxes * np.array([width, height, width, height])
                boxes = torch.tensor(boxes)
                labels = torch.tensor(target["class_labels"])
                post_processed_targets.append({"boxes": boxes, "labels": labels})
        return post_processed_targets

    def collect_predictions(self, predictions, image_sizes):
        """
        Process and collect predictions for evaluation.

        Args:
            predictions (list): List of prediction tuples.
            image_sizes (list): List of image size tensors.

        Returns:
            list: List of processed prediction dictionaries.
        """
        post_processed_predictions = []
        for batch, target_sizes in zip(predictions, image_sizes):
            batch_logits, batch_boxes = batch[1], batch[2]
            output = ModelOutput(
                logits=torch.tensor(batch_logits),
                pred_boxes=torch.tensor(batch_boxes)
            )
            post_processed_output = self.image_processor.post_process_object_detection(
                output, threshold=self.threshold, target_sizes=target_sizes
            )
            post_processed_predictions.extend(post_processed_output)
        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, evaluation_results):
        """
        Compute evaluation metrics.

        Args:
            evaluation_results: Results from the Trainer's evaluation.

        Returns:
            dict: Dictionary of computed metrics.
        """
        predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

        image_sizes = self.collect_image_sizes(targets)
        post_processed_targets = self.collect_targets(targets, image_sizes)
        post_processed_predictions = self.collect_predictions(predictions, image_sizes)

        evaluator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        evaluator.warn_on_many_detections = False
        evaluator.update(post_processed_predictions, post_processed_targets)

        metrics = evaluator.compute()

        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_name = self.id2label[class_id.item()] if self.id2label else class_id.item()
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        return metrics


class PyTorchDetectionDataset(Dataset):
    """
    PyTorch Dataset for object detection tasks.
    """

    def __init__(self, dataset: sv.DetectionDataset, processor, transform: A.Compose = None):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform

    @staticmethod
    def annotations_as_coco(image_id, categories, boxes):
        """
        Convert annotations to COCO format.

        Args:
            image_id (int): Image identifier.
            categories (list): List of category IDs.
            boxes (list): List of bounding boxes.

        Returns:
            dict: Annotations in COCO format.
        """
        annotations = []
        for category, bbox in zip(categories, boxes):
            x1, y1, x2, y2 = bbox
            formatted_annotation = {
                "image_id": image_id,
                "category_id": category,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "iscrowd": 0,
                "area": (x2 - x1) * (y2 - y1),
            }
            annotations.append(formatted_annotation)

        return {
            "image_id": image_id,
            "annotations": annotations,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieve a single data point.

        Args:
            idx (int): Index of the data point.

        Returns:
            dict: Processed data point.
        """
        _, image, annotations = self.dataset[idx]

        # Convert image from BGR to RGB
        image = image[:, :, ::-1]
        boxes = annotations.xyxy
        categories = annotations.class_id

        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                category=categories
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            categories = transformed["category"]

        formatted_annotations = self.annotations_as_coco(
            image_id=idx, categories=categories, boxes=boxes)
        result = self.processor(
            images=image, annotations=formatted_annotations, return_tensors="pt")

        result = {k: v[0] for k, v in result.items()}

        return result


def collate_fn(batch):
    """
    Collate function to combine batch data.

    Args:
        batch (list): List of data points.

    Returns:
        dict: Collated batch.
    """
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an Object Detection Model with Hugging Face Transformers.")

    parser.add_argument(
        "--img_size", type=int, required=True
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save training outputs."
    )
    parser.add_argument(
        "--train_images_directory_path", type=str, required=True,
        help="Path to the training images directory."
    )
    parser.add_argument(
        "--train_annotations_path", type=str, required=True,
        help="Path to the training annotations JSON file."
    )
    parser.add_argument(
        "--val_images_directory_path", type=str, required=True,
        help="Path to the validation images directory."
    )
    parser.add_argument(
        "--val_annotations_path", type=str, required=True,
        help="Path to the validation annotations JSON file."
    )

    return parser.parse_args()


def main(args):
    CHECKPOINT = "PekingU/rtdetr_r50vd_coco_o365"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForObjectDetection.from_pretrained(
        CHECKPOINT,
        id2label=None, 
        label2id=None,  
        anchor_image_size=None,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    processor = AutoImageProcessor.from_pretrained(
        CHECKPOINT,
        do_resize=True,
        size={"width": args.img_size, "height": args.img_size},
    )

    ds_train = sv.DetectionDataset.from_coco(
        images_directory_path=args.train_images_directory_path,
        annotations_path=args.train_annotations_path,
    )
    ds_valid = sv.DetectionDataset.from_coco(
        images_directory_path=args.val_images_directory_path,
        annotations_path=args.val_annotations_path,
    )

    train_augmentation_and_transform = A.Compose(
        [
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category"],
            clip=True,
            min_area=25
        ),
    )

    valid_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category"],
            clip=True,
            min_area=1
        ),
    )

    pytorch_dataset_train = PyTorchDetectionDataset(
        ds_train, processor, transform=train_augmentation_and_transform)
    pytorch_dataset_valid = PyTorchDetectionDataset(
        ds_valid, processor, transform=valid_transform)

    id2label = {id: label for id, label in enumerate(ds_train.classes)}
    label2id = {label: id for id, label in enumerate(ds_train.classes)}

    model.config.id2label = id2label
    model.config.label2id = label2id

    eval_compute_metrics_fn = MAPEvaluator(
        image_processor=processor,
        threshold=0.01,
        id2label=id2label
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=50,
        max_grad_norm=0.1,
        learning_rate=5e-5,
        warmup_steps=300,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        dataloader_num_workers=2,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=4,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=pytorch_dataset_train,
        eval_dataset=pytorch_dataset_valid,
        tokenizer=processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    trainer.train()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
