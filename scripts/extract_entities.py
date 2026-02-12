from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

from src.utils.io import write_json
from src.utils.paths import data_dir
from src.vision.entity_extractor import classify_scenes_places365, detect_objects_yolo


app = typer.Typer(add_completion=False, help="Extract grounded entities (YOLOv8) + scenes (Places365) from frames.")
console = Console()


def _iter_clip_frame_dirs(frames_root: Path) -> list[Path]:
    if not frames_root.exists():
        return []
    return sorted([p for p in frames_root.iterdir() if p.is_dir()])


@app.command()
def run(
    frames_root: Path = typer.Option(None, help="Defaults to DATA_DIR/frames"),
    out_dir: Path = typer.Option(None, help="Defaults to DATA_DIR/entities"),
    overwrite: bool = typer.Option(False, help="Overwrite existing entity JSON."),
    yolo_model: str = typer.Option("yolov8n.pt", help="Ultralytics model name or path."),
    yolo_conf: float = typer.Option(0.25, help="YOLO confidence threshold."),
    yolo_iou: float = typer.Option(0.45, help="YOLO IoU threshold."),
    places_weights: Path = typer.Option(
        None,
        help="Path to Places365 ResNet18 weights. If omitted or missing, Places365 is skipped.",
    ),
    places_top_k: int = typer.Option(3, help="Top-K scenes per frame to aggregate (weighted vote)."),
    device: str = typer.Option(None, help="Device override, e.g. 'cpu', 'cuda:0'."),
):
    run_job(
        frames_root=frames_root,
        out_dir=out_dir,
        overwrite=overwrite,
        yolo_model=yolo_model,
        yolo_conf=yolo_conf,
        yolo_iou=yolo_iou,
        places_weights=places_weights,
        places_top_k=places_top_k,
        device=device,
    )


def run_job(
    *,
    frames_root: Path | None = None,
    out_dir: Path | None = None,
    overwrite: bool = False,
    yolo_model: str = "yolov8n.pt",
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.45,
    places_weights: Path | None = None,
    places_top_k: int = 3,
    device: str | None = None,
):
    frames_root = frames_root or (data_dir() / "frames")
    out_dir = out_dir or (data_dir() / "entities")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Allow env override for weights path
    if places_weights is None:
        p = os.getenv("PLACES365_WEIGHTS")
        if p:
            places_weights = Path(p).expanduser().resolve()

    clip_dirs = _iter_clip_frame_dirs(frames_root)
    console.print(f"[bold]Entity extraction[/bold] clips={len(clip_dirs)} -> {out_dir}")

    with Progress() as progress:
        task = progress.add_task("Extracting", total=len(clip_dirs))
        for d in clip_dirs:
            progress.advance(task)
            clip_id = d.name
            out_json = out_dir / f"{clip_id}.json"
            if out_json.exists() and not overwrite:
                continue

            frame_paths = sorted(d.glob("*.jpg"))
            if not frame_paths:
                continue

            objects: dict[str, dict[str, float]] = {}
            scenes: dict[str, float] = {}

            # YOLO objects (required)
            try:
                objects = detect_objects_yolo(
                    frame_paths=list(frame_paths),
                    model=Path('./models/yolov8n.pt'),
                    conf=float(yolo_conf),
                    iou=float(yolo_iou),
                    device=device,
                )
            except Exception as e:
                # Best-effort: if YOLO fails, still write a record for debuggability.
                console.print(f"[yellow]YOLO failed[/yellow] clip_id={clip_id} err={e}")

            # Places365 scenes (optional)
            if places_weights is not None and places_weights.exists():
                try:
                    scenes = classify_scenes_places365(
                        frame_paths=list(frame_paths),
                        weights_path=places_weights,
                        top_k=int(places_top_k),
                        device=device,
                    )
                except Exception as e:
                    console.print(f"[yellow]Places365 failed[/yellow] clip_id={clip_id} err={e}")

            write_json(
                out_json,
                {
                    "clip_id": clip_id,
                    "total_frames": len(frame_paths),
                    "objects": objects,
                    "scenes": scenes,
                    "models": {
                        "yolo_model": yolo_model,
                        "yolo_conf": float(yolo_conf),
                        "yolo_iou": float(yolo_iou),
                        "places_weights": str(places_weights) if places_weights else None,
                        "places_top_k": int(places_top_k),
                    },
                },
            )


if __name__ == "__main__":
    app()




