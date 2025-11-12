# Drone Detection

This repository contains code and resources for detecting drones using machine learning techniques.
The project aims to identify drones in various environments using image and video data.
This is a beginning of a project that has an aim to be run on edge devices for real-time drone detection.

The model for object detection is based on YOLOv5s and trained on a custom dataset containing images of drones, birds, and fixed objects.
Model was trained using Ultralytics YOLOv8 framework in Google Colab. Data for dataset was collected from roboflow and other open sources.

## Features
- Real-time drone/bird/fixed-wing detection using pre-trained models
- Support for image and video input

## Examples
![Drone Detection Example](examples/video_1.gif)
![Drone Detection Example](examples/video_2.gif)
![Drone Detection Example](examples/video_3.gif)

# Usage

This project includes a script at `src/main.py` to run YOLO inference on images, videos, or a webcam.
Below are common usage examples and helpful flags.

Prerequisites
- Python 3.10+
- Install dependencies (if not already installed):

```bash
# if tou use  UV environment manager
uv sync

python3 -m pip install -r requirements.txt
# If requirements.txt doesn't include everything:
python3 -m pip install ultralytics torch torchvision opencv-python
```

Basic examples

- Live webcam (show annotated frames):

```bash
python3 src/main.py --source 0 --show
```

- Live webcam and save annotated video:

```bash
python3 src/main.py --source 0 --show --save --out output/webcam_pred.mp4
```

- Run on a video file (show):

```bash
python3 src/main.py --source content/video/video_1.mov --show
```

- Run on a video file and save annotated output:

```bash
python3 src/main.py --source content/video/video_1.mov --save --out output/pred_video_1.mp4
```

- Run on a single image (show):

```bash
python3 src/main.py --source content/images/photo.jpg --show
```

- Single image (save result):

```bash
python3 src/main.py --source content/images/photo.jpg --save --out output/photo_pred.jpg
```

Control device, precision and speed

- Force a device (auto-detected by default):

```bash
python3 src/main.py --source 0 --show --device cuda       # use CUDA
python3 src/main.py --source 0 --show --device mps        # use Apple MPS
python3 src/main.py --source 0 --show --device cpu        # force CPU
```

- Use FP16 on CUDA (if supported):

```bash
python3 src/main.py --source 0 --show --device cuda --half
```

- Reduce compute by processing fewer frames (streaming stride):

```bash
python3 src/main.py --source content/video/video_1.mov --save --stride 2 --out output/stride2.mp4
```

Color handling

If annotated colors look off, try forcing how plotted images are interpreted:

```bash
# Treat model plot output as RGB (swap to BGR for OpenCV)
python3 src/main.py --source 0 --show --color-mode rgb

# Treat plot output as already BGR
python3 src/main.py --source 0 --show --color-mode bgr
```

Where things are located
- Default weights (used by the script if you don't pass `--weights`): `models/v1/my_model.pt`
- Example videos: `content/video/`
- Default output path (when not specified): `output/pred.mp4`

Troubleshooting
- Camera not opening on macOS: give Terminal (or your IDE) Camera access in System Settings → Privacy & Security → Camera.
- If OpenCV fails to open the camera, try other device indices (`--source 1`, `--source 2`) or ensure no other app is using the camera.
- If saving fails with a codec error, try changing the `--out` extension (e.g. `.mkv`) or install `ffmpeg`/alternate codecs.
- If the script is slow, try lowering `--imgsz`, increasing `--stride`, or running on GPU with `--device cuda --half`.

If you'd like, I can add a short two-line example to the README that runs a quick benchmark and prints FPS — tell me if you want that added.
