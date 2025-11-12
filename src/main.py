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
import numpy as np
import threading
import queue
import time
from typing import Union

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


# Precompute some colors (BGR)
COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
    (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
    (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236)
]


def select_device() -> str:
    """Return a device string: 'mps' (Apple), 'cuda' or 'cpu'."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO model on video stream (webcam or file)")
    parser.add_argument("--weights", "-w", type=str, default=str(BASE / "../models/v6/dd6_model.pt"),
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
    parser.add_argument("--stride", type=int, default=5, help="Video stride (skip frames) when using streaming predict")
    parser.add_argument("--skip-frames", type=int, default=0, help="Skip N frames between inferences in fallback mode (0 = no skip)")
    parser.add_argument("--max-queue-size", type=int, default=5, help="Max queue size for threaded pipeline")
    parser.add_argument("--device", type=str, default=None, help="Device to use: 'cpu','cuda','mps' (auto if not set)")
    parser.add_argument("--half", action="store_true", help="Use FP16 where supported (CUDA only)")
    parser.add_argument("--color-mode", type=str, default="auto", choices=["auto","rgb","bgr"],
                        help="Color handling for plotted image: 'auto' detect, 'rgb' treat plot() as RGB, 'bgr' treat plot() as BGR")
    return parser.parse_args()


def open_video_capture(source: Union[str, int]):
    """Open an OpenCV VideoCapture for a source, raising on failure."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap


def ensure_bgr(img: np.ndarray, color_mode: str = "auto") -> np.ndarray:
    """Ensure image is BGR uint8 suitable for OpenCV display.

    color_mode: 'auto'|'rgb'|'bgr'
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if color_mode == "bgr":
        return img
    if color_mode == "rgb":
        # swap channels
        return img[..., ::-1]
    # auto heuristic
    try:
        avg_b = float(img[..., 0].mean())
        avg_r = float(img[..., 2].mean())
        if avg_b > avg_r * 1.2:
            return img
        return img[..., ::-1]
    except Exception:
        return img


def draw_detections_cv2(img: np.ndarray, res, names, conf_thres: float = 0.0) -> np.ndarray:
    """Draw bounding boxes and labels from a Results object onto img (BGR).

    This is much faster than res.plot() which uses PIL/matplotlib.
    """
    out = img.copy()
    boxes_obj = getattr(res, "boxes", None)
    if boxes_obj is None:
        return out
    # boxes_obj.xyxy, boxes_obj.conf, boxes_obj.cls are torch tensors in most versions
    try:
        xyxy = boxes_obj.xyxy.cpu().numpy()
    except Exception:
        try:
            xyxy = np.array(boxes_obj.xyxy)
        except Exception:
            return out
    try:
        confs = boxes_obj.conf.cpu().numpy()
    except Exception:
        confs = None
    try:
        cls = boxes_obj.cls.cpu().numpy().astype(int)
    except Exception:
        cls = None

    for i, box in enumerate(xyxy):
        if confs is not None and confs[i] < conf_thres:
            continue
        x1, y1, x2, y2 = map(int, box[:4])
        c = int(cls[i]) if cls is not None else 0
        label = None
        try:
            if isinstance(names, dict):
                label = names.get(c, str(c))
            else:
                label = names[c] if names and c < len(names) else str(c)
        except Exception:
            label = str(c)
        color = COLORS[c % len(COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {confs[i]:.2f}" if confs is not None else label
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(out, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


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
    names = getattr(model, 'names', None)

    # Warmup: run a tiny inference to initialize model & CUDA kernels (if applicable)
    try:
        warmup_img = np.zeros((min(64, args.imgsz), min(64, args.imgsz), 3), dtype=np.uint8)
        _ = model.predict(source=warmup_img, imgsz=args.imgsz, conf=0.001, device=device, half=half)
    except Exception:
        pass

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
                                    device=device, stream=True, half=half, vid_stride=args.stride)
    except Exception as e:
        print("Streaming predict failed or not supported by this version of ultralytics:", e)
        use_streaming = False
        results_gen = None

    try:
        if use_streaming and results_gen is not None:
            print("Using streaming inference from ultralytics (generator).")
            for res in results_gen:
                # Prefer drawing on the original frame using OpenCV for speed
                orig = getattr(res, "orig_img", None)
                if orig is not None:
                    try:
                        img_bgr = ensure_bgr(orig, "bgr")
                        img_bgr = draw_detections_cv2(img_bgr, res, names, conf_thres=args.conf)
                    except Exception:
                        continue
                else:
                    # Fall back to plotting then convert
                    try:
                        img = res.plot()
                        img_bgr = ensure_bgr(img, args.color_mode)
                        img_bgr = draw_detections_cv2(img_bgr, res, names, conf_thres=args.conf)
                    except Exception:
                        continue

                # final sanity: ensure dtype & channel order suitable for OpenCV (BGR uint8)
                if isinstance(img_bgr, np.ndarray) and img_bgr.dtype != np.uint8:
                    img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

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
            # Fallback: threaded pipeline (capture -> inference -> display/write) for higher throughput
            print("Falling back to frame-by-frame inference using threaded pipeline.")
            cap = open_video_capture(src)
            fps_read = cap.get(cv2.CAP_PROP_FPS)
            fps = fps_read if fps_read and fps_read > 1 else fps

            frames_q = queue.Queue(maxsize=args.max_queue_size)
            results_q = queue.Queue(maxsize=args.max_queue_size)

            stop_event = threading.Event()

            def capture_thread_fn():
                idx = 0
                while not stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # optionally skip frames (process every skip+1 frames)
                    if args.skip_frames > 0 and (idx % (args.skip_frames + 1)) != 0:
                        idx += 1
                        continue
                    try:
                        frames_q.put((idx, frame), timeout=0.5)
                    except queue.Full:
                        # drop frame if queue is full
                        pass
                    idx += 1
                # signal end
                try:
                    frames_q.put(None, timeout=0.5)
                except Exception:
                    pass

            def inference_thread_fn():
                while True:
                    item = frames_q.get()
                    if item is None:
                        break
                    idx, frame = item
                    # run model.predict on the frame
                    try:
                        results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                                device=device, half=half)
                        res = results[0] if isinstance(results, (list, tuple)) else results
                        # Prefer drawing on original frame for speed
                        try:
                            orig = getattr(res, "orig_img", None)
                            if orig is not None:
                                img_bgr = ensure_bgr(orig, "bgr")
                                img_bgr = draw_detections_cv2(img_bgr, res, names, conf_thres=args.conf)
                            else:
                                img = res.plot()
                                img_bgr = ensure_bgr(img, args.color_mode)
                                img_bgr = draw_detections_cv2(img_bgr, res, names, conf_thres=args.conf)

                            try:
                                results_q.put((idx, img_bgr), timeout=0.5)
                            except queue.Full:
                                pass
                        except Exception:
                            # on error, skip this frame
                            pass
                    except Exception:
                        # In case of inference error, continue
                        pass

                # signal end
                try:
                    results_q.put(None, timeout=0.5)
                except Exception:
                    pass

            # start threads
            cap_thread = threading.Thread(target=capture_thread_fn, daemon=True)
            inf_thread = threading.Thread(target=inference_thread_fn, daemon=True)
            cap_thread.start()
            inf_thread.start()

            # consume results and display/write
            while True:
                item = results_q.get()
                if item is None:
                    break
                idx, img_bgr = item

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
                        stop_event.set()
                        break

                if args.save and writer is not None:
                    writer.write(img_bgr)

            # cleanup
            stop_event.set()
            cap.release()
            cap_thread.join(timeout=1.0)
            inf_thread.join(timeout=1.0)
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
