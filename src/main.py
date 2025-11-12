from ultralytics import YOLO
from pathlib import Path
import torch

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

weights = BASE / "../model/v1/my_model.pt"
source  = BASE / "../content/video/video_0.mov"
outdir  = BASE / "../output"

# Device auto select Apple Silicon → 'mps', or 'cuda' or 'cpu'
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
half = (device == "cuda")  # half працює лише на CUDA

model = YOLO(str(weights))
model.predict(
    source=str(source),
    imgsz=640,
    conf=0.3,
    iou=0.7,
    device=device,
    save=True,
    project=str(outdir),
    name="pred",
    vid_stride=1,
    half=half,
)
