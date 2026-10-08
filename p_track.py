#!/usr/bin/env python3
"""
p_track.py - pedestrian detection + tracking test with Ultralytics YOLO.

Each person gets a box with a persistent ID, and a short trail showing
where they've walked. Press q (or Esc) in the window to quit.

Examples:
  python p_track.py                      # camera 0
  python p_track.py --source 2           # camera 2 (python cameras.py lists them)
  python p_track.py --source walk.mp4    # a video file
  python p_track.py --imgsz 256          # faster, less accurate
  python p_track.py --source walk.mp4 --save out.mp4 --no-show
  python p_track.py --fullscreen         # demo mode
  python p_track.py --window-scale 1.5   # window 1.5x the camera size
"""

import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np

from cameras import add_camera_args, open_frames
from models import add_device_args, load_model

PERSON = 0  # COCO class id for "person"
WINDOW = "pedestrian tracking (q to quit)"


def parse_args():
    p = argparse.ArgumentParser(description="YOLO pedestrian tracking test")
    add_camera_args(p)
    p.add_argument("--model", default="yolo26n.pt",
                   help="model weights; downloaded automatically on first run (default: yolo26n.pt)")
    p.add_argument("--imgsz", type=int, default=320,
                   help="inference size; smaller = faster (default: 320)")
    p.add_argument("--conf", type=float, default=0.35, help="min confidence (default: 0.35)")
    add_device_args(p)
    p.add_argument("--tracker", default="bytetrack.yaml",
                   help="bytetrack.yaml (fast) or botsort.yaml (default: bytetrack.yaml)")
    p.add_argument("--save", default=None, help="optional output video path, e.g. out.mp4")
    p.add_argument("--no-show", action="store_true", help="don't open a window")
    p.add_argument("--window-scale", type=float, default=2.0,
                   help="starting window size as a multiple of the camera image; "
                        "drag the corner to resize (default: 2)")
    p.add_argument("--fullscreen", action="store_true", help="show the window full screen")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run until end)")
    return p.parse_args()


def colour_for(track_id):
    """Stable, distinct-ish colour per track ID."""
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(c) for c in rng.integers(80, 256, size=3))


def main():
    args = parse_args()
    model = load_model(args, track=True, tracker=args.tracker)
    frames = open_frames(args)

    trails = defaultdict(lambda: deque(maxlen=40))  # track id -> recent foot positions
    seen_ids = set()
    writer = None
    fps = 0.0
    t_prev = time.perf_counter()
    n = 0

    for frame in frames:
        n += 1

        # persist=True keeps tracker state between calls, so IDs carry across frames
        results = model.track(frame, persist=True, classes=[PERSON], conf=args.conf,
                              imgsz=args.imgsz, tracker=args.tracker, device=args.device, verbose=False)
        boxes = results[0].boxes

        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, confs):
                seen_ids.add(tid)
                col = colour_for(tid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"#{tid} {c:.2f}", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                # bottom-centre of the box approximates where the person stands
                trails[tid].append(((x1 + x2) // 2, y2))
                pts = np.array(trails[tid], dtype=np.int32)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, col, 2)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        people_now = 0 if boxes is None else len(boxes)
        cv2.putText(frame, f"{fps:4.1f} fps  people: {people_now}  unique IDs: {len(seen_ids)}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
            writer.write(frame)

        if not args.no_show:
            if n == 1:
                # WINDOW_NORMAL scales the image (and everything drawn on it) to the window
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                if args.fullscreen:
                    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    h, w = frame.shape[:2]
                    cv2.resizeWindow(WINDOW, int(w * args.window_scale), int(h * args.window_scale))
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        elif n % 30 == 0:
            print(f"frame {n}: {fps:.1f} fps, {people_now} people, {len(seen_ids)} unique IDs")

        if args.max_frames and n >= args.max_frames:
            break

    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done: {n} frames, {len(seen_ids)} unique people tracked.")


if __name__ == "__main__":
    main()
