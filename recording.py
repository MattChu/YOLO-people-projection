"""
recording.py - start/stop video recordings with keys, shared by calibrate.py,
foot_calibrate.py, follow.py and track_osc.py.

  c  camera window: the camera image with everything drawn on it
  p  projector: exactly what's sent to the projector
  f  full screen: the laptop screen, all windows as you see them

Each key starts a recording, and the same key stops it. They're independent, so
any mix can run at once. Files go in --record-dir (default: recordings/), named
by time, script and what's recorded, e.g. 2026-10-05_201502_follow_camera.mp4.
Camera and projector recordings play back at real speed: frames are repeated or
dropped to keep a steady --record-fps, whatever speed the script runs at.

Files are H.264 .mp4, which plays almost anywhere (browsers, phones, editors).
Each recording is encoded by its own ffmpeg process, the one bundled with the
imageio-ffmpeg package, so encoding stays out of the tracking loop. The full
screen is captured by ffmpeg directly: the main screen (the one at the top
left of the desktop), with x11grab on Linux (X11, not Wayland), gdigrab on
Windows and avfoundation on macOS (which asks for Screen Recording permission
the first time). Only the Linux one has been tried.

A red "REC" note in the camera window shows what's recording. It's drawn after
the camera frame is recorded, so it never appears in the camera recording.
"""

import atexit
import os
import re
import subprocess
import sys
import time

import cv2

KEYS = {ord("c"): "camera", ord("p"): "projector", ord("f"): "screen"}
# veryfast: light enough to run beside YOLO; yuv420p: what players expect;
# faststart: the file can start playing before it's fully downloaded
H264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def add_record_args(p):
    """Add the --record-dir / --record-fps / --screen-scale options to a parser."""
    p.add_argument("--record-dir", default="recordings", help="where recordings go (default: recordings)")
    p.add_argument("--record-fps", type=float, default=15, help="recording frame rate (default: 15)")
    p.add_argument("--screen-scale", type=float, default=0.5,
                   help="full-screen recording size as a multiple of the screen's; smaller is "
                        "lighter on the CPU (default: 0.5)")


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
    except ImportError:
        raise RuntimeError("recording needs imageio-ffmpeg: pip install imageio-ffmpeg")
    return imageio_ffmpeg.get_ffmpeg_exe()


class Ffmpeg:
    """An ffmpeg process writing one file. stop() asks it to finish the file."""

    def __init__(self, args, path):
        self.path = path
        self.proc = subprocess.Popen([ffmpeg_exe(), "-loglevel", "error", "-y", *args, path],
                                     stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def stop(self, finish=None):
        """End the input (or send `finish` on stdin), wait for the file to be
        written, and raise if ffmpeg failed."""
        try:
            if finish:
                self.proc.stdin.write(finish)
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        try:
            _, err = self.proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            raise RuntimeError(f"ffmpeg didn't finish {self.path}")
        if self.proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {err.decode(errors='replace').strip()[-300:]}")


class PacedWriter:
    """Video file written at a steady frame rate from frames that arrive
    unevenly: each frame is repeated until the next one arrives, so playback
    runs at real speed. ffmpeg starts on the first frame, when the size is known."""

    def __init__(self, path, fps):
        self.path, self.fps = path, fps
        self.ffmpeg = None
        self.start = None
        self.written = 0

    def add(self, img, now=None):
        now = time.perf_counter() if now is None else now
        if self.ffmpeg is None:
            h, w = img.shape[:2]
            self.ffmpeg = Ffmpeg(["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(self.fps),
                                  "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", *H264], self.path)
            self.start = now
        due = int((now - self.start) * self.fps) + 1  # frames that should exist by now
        data = img.tobytes()
        try:
            while self.written < due:
                self.ffmpeg.proc.stdin.write(data)
                self.written += 1
        except BrokenPipeError:
            self.ffmpeg.stop()  # raises with ffmpeg's own error message
            raise RuntimeError("ffmpeg stopped unexpectedly")

    def close(self):
        """Finish the file. Returns its length in seconds."""
        if self.ffmpeg is not None:
            self.ffmpeg.stop()
        return self.written / self.fps


def laptop_screen(xrandr_monitors):
    """(width, height) of the screen at the desktop's top-left corner - the
    laptop's own (the projector sits to its right) - from `xrandr --listmonitors`."""
    for w, h, x, y in re.findall(r"(\d+)/\d+x(\d+)/\d+\+(\d+)\+(\d+)", xrandr_monitors):
        if x == y == "0":
            return int(w), int(h)
    raise RuntimeError("couldn't find the laptop screen in xrandr's monitor list")


def screen_input(fps):
    """ffmpeg's input options for capturing the main screen on this system."""
    if sys.platform.startswith("linux"):
        display = os.environ.get("DISPLAY")
        if not display:
            raise RuntimeError("full-screen recording needs an X11 display (DISPLAY isn't set; Wayland isn't supported)")
        try:
            monitors = subprocess.run(["xrandr", "--listmonitors"], capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError) as e:
            raise RuntimeError(f"couldn't run xrandr: {e}")
        w, h = laptop_screen(monitors)
        return ["-f", "x11grab", "-framerate", str(fps), "-video_size", f"{w}x{h}", "-i", f"{display}+0,0"]
    if sys.platform == "win32":
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()  # real pixels, not scaled ones
        w, h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)  # the main screen
        return ["-f", "gdigrab", "-framerate", str(fps), "-offset_x", "0", "-offset_y", "0",
                "-video_size", f"{w}x{h}", "-i", "desktop"]
    if sys.platform == "darwin":
        return ["-f", "avfoundation", "-framerate", str(fps), "-capture_cursor", "1", "-i", "Capture screen 0:none"]
    raise RuntimeError(f"full-screen recording isn't set up for {sys.platform}")


class ScreenRecording:
    """ffmpeg capturing the main screen itself, at a steady frame rate."""

    def __init__(self, path, fps, scale):
        source = screen_input(fps)
        self.path, self.start = path, time.perf_counter()
        size = f"trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2"  # H.264 needs even sizes
        self.ffmpeg = Ffmpeg([*source, "-vf", f"scale={size}", *H264], path)

    def close(self):
        seconds = time.perf_counter() - self.start
        self.ffmpeg.stop(finish=b"q")  # q = ffmpeg's own "stop recording" key
        return seconds


class Recorder:
    """The recordings for one script run. Per frame: record(), then overlay(),
    then handle_key() with the key from cv2.waitKey()."""

    def __init__(self, args, script, kinds=("camera", "projector", "screen")):
        """kinds: the recordings this script offers (track_osc.py has no projector image)."""
        self.dir, self.fps, self.screen_scale = args.record_dir, args.record_fps, args.screen_scale
        self.script = script
        self.keys = {k: name for k, name in KEYS.items() if name in kinds}
        self.active = {}  # name -> PacedWriter or ScreenRecording
        atexit.register(self.close)  # an unfinished .mp4 won't play, so finish them however the script ends
        names = {"camera": "camera", "projector": "projector", "screen": "full screen"}
        print("Recording keys: " + " / ".join(chr(k) for k in self.keys) + " = record "
              + " / ".join(names[n] for n in self.keys.values()))

    def handle_key(self, key):
        """Start or stop the recording for this key. True if the key was one of ours."""
        name = self.keys.get(key)
        if name is None:
            return False
        if name in self.active:
            self._stop(name)
        else:
            self._start(name)
        return True

    def record(self, name, img):
        """Add a frame to the camera or projector recording, if it's running."""
        rec = self.active.get(name)
        if isinstance(rec, PacedWriter):
            try:
                rec.add(img)
            except RuntimeError as e:
                del self.active[name]
                print(f"Recording {name} stopped: {e}")

    def overlay(self, img):
        """Red REC note at the bottom left of the camera view (call after record())."""
        if not self.active:
            return
        text = "REC " + " + ".join(self.active)
        y = img.shape[0] - 12
        cv2.circle(img, (18, y - 6), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(img, text, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, text, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    def close(self):
        for name in list(self.active):
            self._stop(name)

    def _start(self, name):
        os.makedirs(self.dir, exist_ok=True)
        path = os.path.join(self.dir, f"{time.strftime('%Y-%m-%d_%H%M%S')}_{self.script}_{name}.mp4")
        try:
            if name == "screen":
                self.active[name] = ScreenRecording(path, self.fps, self.screen_scale)
            else:
                ffmpeg_exe()  # fail now, not on the first frame
                self.active[name] = PacedWriter(path, self.fps)
        except RuntimeError as e:
            print(f"Can't record {name}: {e}")
            return
        print(f"Recording {name} -> {path}")

    def _stop(self, name):
        rec = self.active.pop(name)
        try:
            seconds = rec.close()
            print(f"Stopped recording {name}: {rec.path} ({seconds:.1f} s)")
        except RuntimeError as e:
            print(f"Stopped recording {name}, but {e}")
