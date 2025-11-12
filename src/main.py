#!/usr/bin/env python3
"""Run YOLO model on a video stream (webcam or file).

Features:
- Auto-select device (mps / cuda / cpu)
- Support webcam (0) or video file
- Live display with OpenCV
- Optional saving of annotated video
- CLI args for weights, source, confidence, etc.
"""
from ultralytics import YOLO
from pathlib import Path
import torch
import argparse
import cv2
from typing import Union

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def select_device() -> str:
    """Return a device string: 'mps' (Apple), 'cuda' or 'cpu'."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO model on video stream (webcam or file)")
    parser.add_argument("--weights", "-w", type=str, default=str(BASE / "../models/v1/my_model.pt"),
                        help="Path to model weights (.pt)")
    parser.add_argument("--source", "-s", type=str, default="0",
                        help="Video source: integer for webcam (0), or file path")
    parser.add_argument("--out", "-o", type=str, default=str(BASE / "../output/pred.mp4"),
                        help="Output video path when --save is used")
    parser.add_argument("--show", action="store_true", help="Show live annotated video window")
    parser.add_argument("--save", action="store_true", help="Save annotated output to --out")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default=None, help="Device to use: 'cpu','cuda','mps' (auto if not set)")
    parser.add_argument("--half", action="store_true", help="Use FP16 where supported (CUDA only)")
    return parser.parse_args()


def open_video_capture(source: Union[str, int]):
    """Open an OpenCV VideoCapture for a source, raising on failure."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap


def main():
    args = parse_args()

    # decide device and FP precision
    device = args.device or select_device()
    half = args.half or (device == "cuda")

    weights = args.weights

    # interpret source ("0" -> webcam 0)
    src_input = args.source
    try:
        src = int(src_input)
    except Exception:
        src = src_input

    # Validate weights
    w_path = Path(weights)
    if not w_path.exists():
        print(f"Warning: weights file '{w_path}' not found. Continuing and letting ultralytics handle loading.")

    print(f"Loading model: {weights} on device={device} half={half}")
    model = YOLO(str(weights))

    # Prepare saving
    writer = None
    out_path = Path(args.out)
    if args.save:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fps = 30.0
    frame_size = None

    # Try to use ultralytics streaming predict (yields Results)
    use_streaming = True
    try:
        results_gen = model.predict(source=src, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                    device=device, stream=True, half=half)
    except Exception as e:
        print("Streaming predict failed or not supported by this version of ultralytics:", e)
        use_streaming = False
        results_gen = None

    try:
        if use_streaming and results_gen is not None:
            print("Using streaming inference from ultralytics (generator).")
            for res in results_gen:
                # res is a Results object
                try:
                    img = res.plot()  # typically RGB
                except Exception:
                    # fallback to original image
                    img = getattr(res, "orig_img", None)
                    if img is None:
                        continue
                # Convert to BGR for OpenCV display
                try:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                except Exception:
                    img_bgr = img

                if frame_size is None:
                    h, w = img_bgr.shape[:2]
                    frame_size = (w, h)
                    if args.save:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)

                if args.show:
                    cv2.imshow("YOLO Stream", img_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("User requested exit (q pressed)")
                        break

                if args.save and writer is not None:
                    writer.write(img_bgr)

        else:
            # Fallback: read frames with OpenCV and call model.predict on each frame
            print("Falling back to frame-by-frame inference using OpenCV capture.")
            cap = open_video_capture(src)
            fps_read = cap.get(cv2.CAP_PROP_FPS)
            fps = fps_read if fps_read and fps_read > 1 else fps

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of stream or cannot read frame.")
                    break

                # Run inference on this frame (ultralytics accepts numpy arrays)
                results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                        device=device, half=half)
                # results may be list-like
                res = results[0] if isinstance(results, (list, tuple)) else results
                try:
                    img = res.plot()
                except Exception:
                    img = getattr(res, "orig_img", frame)
                try:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                except Exception:
                    img_bgr = img

                if frame_size is None:
                    h, w = img_bgr.shape[:2]
                    frame_size = (w, h)
                    if args.save:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)

                if args.show:
                    cv2.imshow("YOLO Stream", img_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("User requested exit (q pressed)")
                        break

                if args.save and writer is not None:
                    writer.write(img_bgr)

            cap.release()

    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
