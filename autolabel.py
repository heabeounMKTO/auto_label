from utils.torch_utils import select_device
import cv2
from pathlib import Path
import os
import configparser
import json
import base64
import tqdm
from rich import print
from rich.console import Console
from rich.table import Table
from ultralytics import YOLO

# read global config file
config = configparser.ConfigParser()
config.read("auto-labelcfg.ini")

MODEL_PATH = os.path.join("detect-models", config["MODEL"]["MODEL_NAME"])
CUDA_DEVICE = select_device(str(config["DEVICE_SETTINGS"]["CUDA_DEVICE"]))
CONFIDENCE_THRESH = float(config["INFERENCE_CONFIG"]["CONFIDENCE"])
ext = [".jpeg", ".jpg", ".png", ".webp"]


def _convertImg2b64(imgpath):
    with open(imgpath, "rb") as img_file:
        imgstring = base64.b64encode(img_file.read())
    return imgstring


def _bbox_iou(a: list, b: list) -> float:
    """
    Compute IoU between two LabelMe rectangle shapes.
    Each shape has points = [[x1,y1],[x2,y2]].
    """
    ax1, ay1 = a[0]
    ax2, ay2 = a[1]
    bx1, by1 = b[0]
    bx2, by2 = b[1]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _is_duplicate(new_shape: dict, existing_shapes: list, iou_threshold: float) -> bool:
    """Return True if new_shape overlaps an existing same-label shape above the IoU threshold."""
    for existing in existing_shapes:
        if existing.get("label") != new_shape["label"]:
            continue
        if existing.get("shape_type") != "rectangle":
            continue
        iou = _bbox_iou(new_shape["points"], existing["points"])
        if iou >= iou_threshold:
            return True
    return False


def _load_label_filter(labels: list[str] | None, labels_file: str | None) -> set[str] | None:
    """
    Build the allowed-labels set.
    labels_file takes priority over the CLI labels list.
    Returns None if no filter is active (allow everything).
    """
    if labels_file:
        p = Path(labels_file)
        if not p.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        allowed = {line.strip() for line in p.read_text().splitlines() if line.strip()}
        print(f"[green]Label filter loaded from file:[/green] {allowed}")
        return allowed

    if labels:
        allowed = set(labels)
        print(f"[green]Label filter from CLI:[/green] {allowed}")
        return allowed

    return None  # no filter — accept all


class AutoLabel:
    def __init__(self, processingFolder, confidence=0.9, iou=0.7):
        self.processingFolder = processingFolder
        self.confidence = confidence
        self.iou = iou
        print("config loaded:")
        print(
            f"Model: {MODEL_PATH}\nCuda Device: {CUDA_DEVICE}\nConfidence: {CONFIDENCE_THRESH}"
        )
        self.labeledimg = 0

    def findAllImageInFolder(self):
        imagePaths = []
        print(f"looking for files in {self.processingFolder}")
        for roots, dirs, files in os.walk(self.processingFolder):
            print(f"found {len(files)} files in {self.processingFolder}")
            for file in files:
                if file.endswith(tuple(ext)):
                    self.labeledimg += 1
                    fullimagepath = os.path.join(self.processingFolder, file)
                    imagePaths.append(fullimagepath)
            return imagePaths

    def Label(
        self,
        use_ultralytics: bool = False,
        append: bool = False,
        iou_threshold: float = 0.5,
        labels: list[str] | None = None,
        labels_file: str | None = None,
    ):
        """
        Run inference and write/update LabelMe JSONs.

        Args:
            append:        If True and a JSON already exists, append new shapes
                           instead of overwriting. Skips duplicates above iou_threshold.
            iou_threshold: IoU above which a new detection is considered a duplicate
                           of an existing same-label shape (only used when append=True).
            labels:        Optional list of class names to keep. Detections outside
                           this set are silently dropped.
            labels_file:   Path to a text file (one label per line) that defines the
                           allowed labels. Takes priority over `labels` if both are given.
        """
        images = self.findAllImageInFolder()
        allowed_labels = _load_label_filter(labels, labels_file)

        load_model = YOLO(MODEL_PATH, verbose=False)
        model_classes = load_model.names

        total_written = 0
        total_skipped_filter = 0
        total_skipped_iou = 0
        total_appended = 0

        for image_pth in tqdm.tqdm(images):
            img = cv2.imread(image_pth)
            results = load_model(img, show=False, verbose=False, conf=CONFIDENCE_THRESH)

            json_path = Path(image_pth).with_suffix(".json")

            # --- Load existing annotation if appending ---
            existing_shapes = []
            base_annotation = None
            if append and json_path.exists():
                with open(json_path, "r") as f:
                    base_annotation = json.load(f)
                existing_shapes = base_annotation.get("shapes", [])

            # --- Collect new detections ---
            new_shapes = []
            for result in results:
                classes = result.boxes.cls.tolist()
                confidence = result.boxes.conf.tolist()
                xyxys = result.boxes.xyxy.tolist()

                for idx, xyxy in enumerate(xyxys):
                    class_name = model_classes[int(classes[idx])]
                    _confidence = confidence[idx]
                    x1, y1, x2, y2 = [int(x) for x in xyxy]

                    # Label filter
                    if allowed_labels is not None and class_name not in allowed_labels:
                        total_skipped_filter += 1
                        continue

                    shape = {
                        "label": class_name,
                        "points": [[x1, y1], [x2, y2]],
                        "group_id": f"{_confidence:.4f}",
                        "shape_type": "rectangle",
                        "flags": {},
                    }

                    # IoU dedup against already-accepted shapes (existing + new so far)
                    if append and _is_duplicate(shape, existing_shapes + new_shapes, iou_threshold):
                        total_skipped_iou += 1
                        continue

                    new_shapes.append(shape)

            # --- Build final annotation ---
            if append and base_annotation is not None:
                base_annotation["shapes"] = existing_shapes + new_shapes
                final_annotation = base_annotation
                total_appended += len(new_shapes)
            else:
                final_annotation = {
                    "version": "5.1.1",
                    "flags": {},
                    "shapes": new_shapes,
                    "imagePath": os.path.basename(image_pth),
                    "imageData": _convertImg2b64(image_pth).decode("utf-8"),
                    "imageHeight": img.shape[0],
                    "imageWidth": img.shape[1],
                }
                total_written += len(new_shapes)

            with open(json_path, "w") as json_output:
                json_output.write(json.dumps(final_annotation, indent=2))

        # --- Summary ---
        console = Console()
        table = Table(title="AutoLabel Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_row("Images processed", str(self.labeledimg))
        table.add_row("Shapes written (new JSONs)", str(total_written))
        table.add_row("Shapes appended (existing JSONs)", str(total_appended))
        table.add_row("Dropped by label filter", str(total_skipped_filter))
        table.add_row("Dropped by IoU dedup", str(total_skipped_iou))
        console.print(table)
