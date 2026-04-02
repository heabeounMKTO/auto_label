import os
from pathlib import Path
from typing import Optional
import typer
from autolabel import AutoLabel
from rich import print

app = typer.Typer()


@app.command()
def main(
    source: str = typer.Option(..., "--source", help="Folder of images to label"),
    use_ultralytics: bool = typer.Option(
        False,
        "--use-ultralytics",
        help="Use Ultralytics YOLO backend for inference.",
    ),
    append: bool = typer.Option(
        False,
        "--append",
        help="If a LabelMe JSON already exists for an image, append new shapes instead of overwriting.",
    ),
    iou_threshold: float = typer.Option(
        0.5,
        "--iou-threshold",
        help="IoU threshold for duplicate suppression when appending (default: 0.5).",
    ),
    labels: Optional[list[str]] = typer.Option(
        None,
        "--labels",
        help="Only keep detections with these class names. Repeat for multiple: --labels foo --labels bar",
    ),
    labels_file: Optional[str] = typer.Option(
        None,
        "--labels-file",
        help="Path to a text file with one allowed label per line. Takes priority over --labels.",
    ),
):
    label = AutoLabel(source)
    label.Label(
        use_ultralytics=use_ultralytics,
        append=append,
        iou_threshold=iou_threshold,
        labels=labels if labels else None,
        labels_file=labels_file,
    )


if __name__ == "__main__":
    app()
