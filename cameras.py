"""
cameras.py - frame sources shared by p_track.py, calibrate.py, foot_calibrate.py,
follow.py and track_osc.py.

Any camera OpenCV can open: a webcam, a USB or capture-card camera, or a
depth camera's colour feed (an Intel RealSense's shows up as an ordinary
camera, for instance), chosen by its number. Or a video file, for testing.

Run it to list the cameras it can find, and pick the number to use with --source:

  python cameras.py              # list them
  python cameras.py --snapshots  # also save a picture from each, camera_N.jpg, to see which is which
"""

import argparse
import math
import threading

import cv2

# What OpenCV's CAP_PROP_AUTO_EXPOSURE means (manual, auto) depends on the
# system's camera layer; elsewhere only the exposure itself is set
AUTO_EXPOSURE = {"V4L2": (1, 3), "DSHOW": (0.25, 0.75)}


def add_camera_args(p):
    """Add the --source / --width / --height / --fps / --fourcc / --exposure options to a parser."""
    p.add_argument("--source", default=None,
                   help="camera number (0, 1...: python cameras.py lists them) or a video file "
                        "(default: 0, or the camera the calibration was made with)")
    p.add_argument("--width", type=int, default=None,
                   help="ask the camera for frames this wide (default: the size it was calibrated "
                        "at, or the camera's own)")
    p.add_argument("--height", type=int, default=None, help="and this high (as --width)")
    p.add_argument("--fps", type=float, default=None,
                   help="ask the camera for this many frames a second (default: the rate it was "
                        "calibrated at, or the camera's own)")
    p.add_argument("--fourcc", default=None,
                   help="ask the camera for this pixel format, e.g. MJPG: many USB cameras only "
                        "give their largest sizes or highest frame rates compressed (default: the "
                        "format it was calibrated with, or the camera's own)")
    p.add_argument("--exposure", type=float, default=None,
                   help="fix the camera's exposure, in its own units (Linux: usually 1-10000, "
                        "in 0.1 ms; Windows: a power of 2 in seconds, e.g. -6 for 1/64 s), "
                        "where the camera and system allow it (default: auto). A bright "
                        "projection in a dark room may need a fixed one: auto exposure "
                        "brightens for the room and washes out the floor")


def source_name(args, default="0"):
    """The chosen camera as one string: a camera number or a video path.
    `default` is used when --source wasn't given."""
    return args.source if args.source is not None else default


def calibrated_camera(args, cam):
    """The source to open for calibrated camera `cam` (its entry in
    calibration.json): the one it was calibrated with, unless --source says
    otherwise. Also asks for the size, frame rate and pixel format it was
    calibrated at, where not given on the command line."""
    calibrated_source = cam.get("source", "0")
    source = source_name(args, default=calibrated_source)
    if source != calibrated_source:
        print(f"Warning: using camera '{source}', but '{args.camera}' was calibrated with "
              f"'{calibrated_source}'. Positions will be wrong unless it's the same view.")
    if args.width is None and args.height is None:
        args.width, args.height = cam["resolution"]
    if args.fps is None:
        args.fps = cam.get("fps")
    if args.fourcc is None:
        args.fourcc = cam.get("fourcc")
    return source


def open_frames(args, source=None):
    """Frame generator for `source` (default: the camera chosen on the command line)."""
    return frames_from_opencv(source or source_name(args), args.width, args.height,
                              getattr(args, "exposure", None), getattr(args, "fps", None),
                              getattr(args, "fourcc", None), getattr(args, "max_fps", None))


def set_exposure(cap, exposure):
    """Fix the camera's exposure (its own units), or with None turn auto
    exposure on (where that's known to work). Returns whether it was accepted."""
    manual, auto = AUTO_EXPOSURE.get(cap.getBackendName(), (None, None))
    if exposure is None:
        if auto is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto)
        return True
    if manual is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual)
    return cap.set(cv2.CAP_PROP_EXPOSURE, exposure)


def frames_from_opencv(source, width=None, height=None, exposure=None, fps=None, fourcc=None,
                       max_fps=None):
    """Yield BGR frames from a camera number or a video file. For a camera,
    ask for this size, frame rate and pixel format where given (the camera
    gives the nearest it can), and fix its exposure."""
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    camera = isinstance(src, int)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera or video: {source}"
                         + (" (python cameras.py lists the cameras it can find)" if camera else ""))
    if camera:
        if fourcc:  # first: it decides which sizes and rates are on offer
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4].ljust(4)))
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
        got_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera {source}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
              + (f" at {got_fps:g} fps" if got_fps > 0 else "") + f" ({cap.getBackendName()})")
        # Cameras keep their exposure setting after we let go of them, so set
        # it either way (a fixed one from an earlier run would otherwise
        # linger), and go back to auto when done
        if not set_exposure(cap, exposure):
            print(f"Camera {source} ({cap.getBackendName()}) didn't accept a fixed exposure: using auto")
            exposure = None
        elif exposure is not None:
            print(f"Camera {source}: exposure fixed at {exposure:g}")
    # at most max_fps frames a second: every `every`th frame the source gives
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    every = max(1, math.ceil(src_fps / max_fps - 1e-6)) if max_fps and src_fps > 0 else 1
    if every > 1:
        print(f"Taking every {ordinal(every)} frame: {src_fps / every:g} a second (at most {max_fps:g})")
    reader = LatestFrame(cap, every) if camera else None
    try:
        while True:
            if reader:
                ok, frame = reader.read()
            else:  # a video file: skip the frames in between
                ok = all(cap.grab() for _ in range(every - 1))
                ok, frame = cap.read() if ok else (False, None)
            if not ok:
                break
            yield frame
    finally:
        if reader:
            reader.stop()
        if camera and exposure is not None:
            set_exposure(cap, None)
        cap.release()


def ordinal(n):
    return {1: "", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


class LatestFrame:
    """Takes a camera's frames on its own thread, and gives out only fresh ones.

    OpenCV queues a few frames from a camera, and read() gives the oldest: when
    the tracker is slower than the camera (YOLO on a CPU), every frame it gets
    is several of its own frames old, and the projection lags behind people by
    up to a second. Here the thread takes every frame off the queue as it
    arrives, but only grabs it (cheap). Once `every` frames have arrived since
    the last one given out, each new one is decoded into a picture (retrieve,
    the costly part) and kept, replacing the one before, so read() gets the
    newest due frame at once if the tracker was busy, or waits for it. Frames
    before then are dropped undecoded: so --max-fps 15 on a 30 fps camera
    takes every 2nd frame, decoding only those."""

    def __init__(self, cap, every=1):
        self.cap = cap
        self.every = every
        self.grabbed = 0                # frames grabbed so far
        self.given = 0                  # the number of the last frame given out
        self.frame, self.number, self.ok = None, 0, True
        self.new = threading.Condition()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        # every call on cap is on this thread: grab and retrieve can't overlap
        while self.running:
            ok = self.cap.grab()
            self.grabbed += 1
            with self.new:
                due = self.grabbed - self.given >= self.every
            frame = None
            if ok and due:
                ok, frame = self.cap.retrieve()
            if due or not ok:
                with self.new:
                    self.ok = ok
                    if ok:
                        self.frame, self.number = frame, self.grabbed
                    self.new.notify()
            if not ok:
                return

    def read(self):
        """The newest due frame, waiting for one if need be: (ok, frame)."""
        with self.new:
            self.new.wait_for(lambda: self.frame is not None or not self.ok)
            frame, self.frame = self.frame, None
            self.given = self.number
        return frame is not None, frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)


def list_cameras(most=10, snapshots=False):
    """Print the cameras OpenCV can open, by number, with the frame size they
    give; with snapshots, save a picture from each as camera_N.jpg."""
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)  # quiet about missing ones
    except AttributeError:
        pass
    found = 0
    for i in range(most):
        cap = cv2.VideoCapture(i)
        frame = None
        if cap.isOpened():
            for _ in range(10):  # a few frames, so auto exposure has settled
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
        if frame is not None:
            found += 1
            h, w = frame.shape[:2]
            saved = ""
            if snapshots:
                cv2.imwrite(f"camera_{i}.jpg", frame)
                saved = f", saved camera_{i}.jpg"
            print(f"  camera {i}: {w}x{h} ({cap.getBackendName()}), average brightness {frame.mean():.0f}{saved}")
        cap.release()
    if not found:
        print("No cameras found")
    else:
        print("Use one with --source N. A camera with several feeds (e.g. depth and colour) "
              "may appear as several numbers: pick the one that's a normal colour picture")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="list the cameras OpenCV can open")
    p.add_argument("--snapshots", action="store_true", help="save a picture from each, camera_N.jpg")
    p.add_argument("--most", type=int, default=10, help="try camera numbers 0 to this - 1 (default: 10)")
    a = p.parse_args()
    list_cameras(a.most, a.snapshots)
