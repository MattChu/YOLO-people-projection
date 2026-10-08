"""
models.py - load YOLO for whatever hardware this machine has.

The code is the same on every machine. What differs is the PyTorch installed
(the CPU or the CUDA build: see requirements.txt) and which model is given
with --model:

  yolo26n-pose.pt          PyTorch: on an NVIDIA GPU with the CUDA build, else the CPU
  yolo26n-pose.engine      TensorRT export, NVIDIA only; build it on the machine
                           that runs it (yolo export model=yolo26n-pose.pt format=engine)
  *_openvino_model/        OpenVINO export, Intel (needs the openvino package)
"""

import numpy as np


def add_device_args(p):
    """Add the --device / --threads options to a parser."""
    p.add_argument("--device", default=None,
                   help="where to run the model: cpu, or 0 for the first NVIDIA GPU "
                        "(default: the GPU if PyTorch can use one, else the CPU)")
    p.add_argument("--threads", type=int, default=1,
                   help="CPU threads for the model, when it runs on the CPU (default: 1). More "
                        "barely speed it up on a laptop but take CPU from the Processing sketch: "
                        "on the dual-core MacBook, 1 thread used 45%% CPU at 15 fps, Ultralytics' "
                        "default of 3 used 130%%, at the same speed")


def load_model(args, track=False, **kwargs):
    """args.model, loaded and run once on a blank frame, so it's on its device
    before the first camera frame arrives and Ultralytics has set its own CPU
    thread count (it does on the first run), which --threads then replaces.
    Pass track=True if the model will be used with model.track(), with the
    same tracker= (the first run sets up the tracker that later ones keep)."""
    import torch
    from ultralytics import YOLO
    model = YOLO(args.model)
    run = model.track if track else model.predict
    run(np.zeros((480, 640, 3), np.uint8), imgsz=args.imgsz, device=args.device, verbose=False, **kwargs)
    torch.set_num_threads(max(1, args.threads))
    device = model.predictor.device
    print(f"Model {args.model} on {device}" + (f", {torch.get_num_threads()} CPU thread(s)" if device.type == "cpu" else ""))
    return model
