# yolo-people-2-the-floor

*[Português](README.pt.md)*

Track people walking on a floor with a camera, know where each of them is
standing in real-world metres, and project images onto the floor exactly under
their feet.

A camera looks down at a taped rectangle on the floor. [YOLO](https://docs.ultralytics.com/)
finds each person and their ankles, the tracker follows them from frame to
frame, and their positions are sent over [OSC](https://opensoundcontrol.stanford.edu/)
to whatever you like: the included [Processing](https://processing.org/)
sketches, TouchDesigner, Max, Pure Data, your own Python, anything that speaks
OSC. A projector calibrated to the same rectangle lets a sketch draw in metres
and have it land on the floor where people are.

**[The illustrated guide](https://mattchu.github.io/yolo-people-2-the-floor/)**
(also [docs/index.html](docs/index.html)) shows the physical setup, the
calibration and how the pieces fit together, with diagrams. Print it to PDF
from your browser if you want a copy on paper.

> **Tested on:** Ubuntu 26.04 (Linux), a 2017 MacBook Pro, a camera at 640×480
> (the colour feed of an Intel RealSense D435, used as an ordinary camera) and a
> 1280×800 projector. The Windows and macOS steps below are written from each
> tool's documentation but haven't been tried yet; please open an issue if
> something doesn't work.

---

## Contents

1. [What you need](#1-what-you-need)
2. [Install the software](#2-install-the-software)
3. [Set up the space](#3-set-up-the-space)
4. [Calibrate](#4-calibrate)
5. [Run it](#5-run-it)
6. [The OSC messages, and using them elsewhere](#6-the-osc-messages-and-using-them-elsewhere)
7. [Make your own Processing sketch](#7-make-your-own-processing-sketch)
8. [The files](#8-the-files)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. What you need

**Hardware**

- **A computer** running Linux, Windows or macOS. Any recent laptop works; an
  NVIDIA graphics card makes the tracking much faster but isn't needed.
- **A projector**, connected as a second screen (extend the desktop, don't
  mirror it), pointing at the floor.
- **A camera** that can see the whole projected area: any camera your
  computer sees as a camera works, from a webcam to a high-end USB or
  capture-card camera, or the colour feed of a depth camera such as an Intel
  RealSense (which shows up as an ordinary camera). Any size and frame rate
  works: a faster camera tracks more smoothly if the computer keeps up, and a
  bigger picture helps find people who are small in it. Mount it **high,
  looking down** at the floor. The more it looks straight down, the better it
  finds people's feet.
- **Masking tape and a tape measure**, to mark a rectangle on the floor.

**Software**

- **Python 3.14** (other recent versions may work, but the pinned versions are
  tested on 3.14).
- **git**, to download this repository.
- **Processing 4** (tested with 4.5.7), for the included sketches. Not needed
  if you'll only use the OSC data in other software.

---

## 2. Install the software

### 2.1 Get Python, git and Processing

| | Python 3.14 | git | Processing 4 |
|---|---|---|---|
| **Linux** (Ubuntu 26.04) | `sudo apt install python3 python3-venv` | `sudo apt install git` | [processing.org/download](https://processing.org/download) |
| **Windows** | [python.org/downloads](https://www.python.org/downloads/) (tick "Add python.exe to PATH") | [git-scm.com](https://git-scm.com/download/win) | [processing.org/download](https://processing.org/download) |
| **macOS** | [python.org/downloads](https://www.python.org/downloads/) | `xcode-select --install` | [processing.org/download](https://processing.org/download) |

### 2.2 Download this repository

```sh
git clone https://github.com/MattChu/yolo-people-2-the-floor.git
cd yolo-people-2-the-floor
```

Every command below is run from inside this folder.

### 2.3 Make a Python environment

The project's Python packages live in their own folder, `yolo-env/`, so they
can't clash with anything else on your computer.

```sh
# Linux and macOS
python3 -m venv yolo-env

# Windows (PowerShell)
py -3.14 -m venv yolo-env
```

### 2.4 Install the packages

Pick the line for your computer. The `requirements-…lock` files pin every
package to an exact, checked file; `requirements.txt` lists just the main
ones, for systems that don't have a lock yet.

**Linux, no NVIDIA graphics card** (there's a lock for this):

```sh
yolo-env/bin/pip install --require-hashes --only-binary :all: \
    --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-linux-cpu.lock
```

**Windows, no NVIDIA graphics card:**

```powershell
yolo-env\Scripts\pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
yolo-env\Scripts\pip install -r requirements.txt
```

**Linux or Windows with an NVIDIA graphics card:** install the CUDA build of
PyTorch first. Run `nvidia-smi` to see which CUDA version your driver
supports: PyTorch 2.14.1 comes as `cu130` (CUDA 13.0, recent drivers) or
`cu126` (CUDA 12.6, older ones).

```sh
yolo-env/bin/pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
yolo-env/bin/pip install -r requirements.txt
```

(on Windows, `yolo-env\Scripts\pip` instead of `yolo-env/bin/pip`)

**macOS:**

```sh
yolo-env/bin/pip install -r requirements.txt
```

**Optional: lock what you installed.** On a system without a lock, you can
make one, so the same exact files install next time:

```sh
yolo-env/bin/python lock_requirements.py               # no NVIDIA card: e.g. requirements-windows-cpu.lock
yolo-env/bin/python lock_requirements.py --torch cu130 # NVIDIA, CUDA 13.0
yolo-env/bin/python lock_requirements.py --torch pypi  # macOS
```

### 2.5 Find your camera's number

Cameras are chosen by number: 0 is usually a laptop's own, and others count
up from there. With the camera plugged in (and the environment active, see
2.6):

```sh
python cameras.py              # list the cameras it can find
python cameras.py --snapshots  # also save a picture from each: camera_0.jpg, camera_1.jpg...
```

On a laptop with a USB camera plugged in, it prints something like:

```
  camera 0: 1280x720 (V4L2), average brightness 104
  camera 2: 1920x1080 (V4L2), average brightness 97
Use one with --source N. ...
```

Here camera 0 is the laptop's own and camera 2 is the USB camera, so you'd use
`--source 2`. The size is what the camera gives by default (you can ask for
others, see 4.1); in brackets is how the system talks to it (V4L2 on Linux,
MSMF or DSHOW on Windows, AVFOUNDATION on macOS). Numbers can skip: some
devices aren't cameras. If you're not sure which is which, look at the
`camera_N.jpg` pictures. A camera with several feeds (a depth camera's depth,
infrared and colour, for instance) appears as several numbers; pick the
normal colour picture.

**A camera over HDMI** (a DSLR, mirrorless camera or camcorder) works through
an HDMI-to-USB capture device, which shows up here as a camera like any
other. Turn on the camera's clean HDMI output, so its on-screen information
isn't in the picture, and turn off its auto power-off.

The first time, your system may ask for permission to use the camera: on
macOS, allow your terminal in **System Settings → Privacy & Security →
Camera**; on Windows, turn on **Settings → Privacy & security → Camera → Let
desktop apps access your camera**.

### 2.6 Using the environment

Each time you open a terminal to run the scripts, switch to the project's
environment first:

```sh
source yolo-env/bin/activate        # Linux and macOS
yolo-env\Scripts\Activate.ps1       # Windows PowerShell
```

Your prompt then starts with `(yolo-env)`, and `python` means the project's
Python. (On Windows, if PowerShell refuses to run the activate script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.) Instead of
activating, you can also run the environment's Python directly, e.g.
`yolo-env/bin/python track_osc.py`. VS Code does it for you once you choose
`yolo-env` with **Python: Select Interpreter**.

Check everything is installed by running the tests:

```sh
python test_osc.py
python test_calibration.py
python test_follow.py
```

The YOLO model (`yolo26n-pose.pt`, about 8 MB) downloads by itself the first
time a script needs it.

---

## 3. Set up the space

The [guide](https://mattchu.github.io/yolo-people-2-the-floor/#setup) has
diagrams of all of this.

1. **Aim the projector** at the floor. It doesn't need to be square on: the
   calibration corrects for any angle.
2. **Connect it as a second screen**, extending the desktop. Note its
   **resolution** and its **position** on the desktop, in pixels, from your
   display settings. For example, a 1280×800 projector placed to the right of
   a 1920-pixel-wide laptop screen is at `1920,0`. On Linux,
   `xrandr --listmonitors` shows both, e.g. `1280/339x800/212+2560+0` is
   1280×800 at `2560,0`. On Windows, set the projector's display scaling to
   100%, so its pixels are real pixels.
3. **Mount the camera** high, looking down, so it sees the whole projected
   area with a little room to spare.
4. **Tape a rectangle** on the floor, **inside** the projected area and in view
   of the camera: four small marks at its corners are enough. **Measure its
   width and height in metres.**
5. **Number the corners in your head:** mark 1 is the origin (0, 0), mark 2 is
   along the width (W, 0), mark 3 is the far corner (W, H), and mark 4 is
   (0, H). Everything (OSC positions, sketches) is measured in metres from
   mark 1, x along the width and y along the height.

---

## 4. Calibrate

Calibration works out how the camera, the projector and the floor line up,
and saves it in `calibration.json`. **Redo it whenever the camera or
projector moves.** Calibrate with the room **as dark as you can**.

### 4.1 `calibrate.py`: camera, projector and floor

```sh
python calibrate.py --source 2 --proj-pos 1920,0 --proj-size 1280x800 --floor-size 2.4x1.4
```

Use your camera's number (2.5), your projector's position and resolution, and
your rectangle's width × height in metres. The camera gives its own size and
frame rate unless you ask for others with `--width`, `--height` and `--fps`;
many USB cameras only give their largest sizes or fastest rates as compressed
frames, so add `--fourcc MJPG` for those. Some examples:

```sh
# camera 2, at the size and rate it gives by default
python calibrate.py --source 2 --proj-pos 1920,0 --proj-size 1280x800 --floor-size 2.4x1.4

# a USB camera at 1920x1080 and 60 frames a second, as compressed frames
python calibrate.py --source 2 --width 1920 --height 1080 --fps 60 --fourcc MJPG \
    --proj-pos 1920,0 --proj-size 1280x800 --floor-size 2.4x1.4

# a camera over HDMI, through capture device 1 (set its exposure on the camera itself)
python calibrate.py --source 1 --width 1920 --height 1080 --fourcc MJPG \
    --proj-pos 1920,0 --proj-size 1280x800 --floor-size 2.4x1.4

# the markers look washed out: fix the exposure (Linux units; see below)
python calibrate.py --source 2 --exposure 80 --proj-pos 1920,0 --proj-size 1280x800 --floor-size 2.4x1.4
```

The camera number and size are saved with the calibration, so the scripts
after this (`foot_calibrate.py`, `follow.py`, `track_osc.py`) use the same
camera at the same size without being told: no `--source` needed. (Give
`--exposure` again if you used it, or set the camera's exposure once on the
camera itself.)

Then, with the keys in the camera window:

1. **Markers.** The projector shows a grid of square markers. Keep the floor
   clear and press **SPACE**. They're found over a few frames, and give the
   camera-to-projector mapping. If too few are found, see
   [Troubleshooting](#9-troubleshooting).
2. **Floor.** The projector turns white. **Click the four tape marks** in the
   camera view, in order 1, 2, 3, 4 (**u** undoes a click).
3. **Verify.** The projector draws a grid in floor coordinates. Its outline
   should sit on your tape. **s** saves, **r** starts again, **q** quits
   without saving.

**If the markers look washed out** (white blobs rather than crisp black and
white squares), fix the camera's exposure: in a dark room, automatic exposure
brightens the picture for the room and overexposes the bright projection.
`--exposure` takes the camera's own units, which depend on the system: on
Linux usually tenths of a millisecond (try `--exposure 80`, then 40 or 160),
on Windows a power of 2 in seconds (try `--exposure -6`, then -7 or -5). Not
every camera, or every system, lets programs set it.

### 4.2 `foot_calibrate.py`: where people's feet land

YOLO sees ankles, which are a little above and behind where a foot touches
the floor. This step measures that and corrects for it.

```sh
python foot_calibrate.py
```

The projector shows numbered targets, one at a time. **Stand with the cross
between your feet** and hold still: after a moment it's captured and the next
one lights up, so one person can do it alone. At the end, **s** saves. (SPACE
captures now, **n** skips a target, **b** goes back one.)

### 4.3 `follow.py`: check it works

```sh
python follow.py
```

A circle is projected under each person's feet. Walk around: it should follow
you. **g** toggles a floor grid, **q** quits.

---

## 5. Run it

In one terminal, start the tracker. It sends everyone's position over OSC:

```sh
python track_osc.py --show        # --show also opens a camera window, to check it
```

Then run a sketch. Open one of these in Processing and press Run:

- `processing/template/template.pde`: a circle at everyone's feet. The
  starting point for your own.
- `processing/grid_peeps/grid_peeps.pde`: lines through everyone, with
  swirling logos between people (press **m** for its other modes: a trail
  behind each person, and a floor full of logos that part round them).
- `processing/pong/pong.pde`: two people play pong the length of the floor.

The sketch goes full screen on the projector if it's connected, or opens a
test window if not.

**Without a camera** (or people), imaginary people wander the floor:

```sh
python track_osc.py --simulate 3                     # three imaginary people, no camera
python track_osc.py --simulate 2 --with-camera --show # you, and two imaginary people
```

In the sketches, holding the mouse button down also makes a test person at
the mouse.

**Useful options** (`python track_osc.py --help` lists them all):

| Option | What it does |
|---|---|
| `--show` | camera window with detections and tracks drawn on |
| `--osc-host 192.168.1.20` | send to another computer (default: this one) |
| `--osc-port 12000` | the OSC port (default 12000) |
| `--max-fps 15` | track at most this many frames a second (default: every frame the camera gives); lower leaves more CPU for a sketch on the same computer |
| `--imgsz 320` | YOLO's image size; smaller is faster, larger finds smaller people |
| `--device cpu` / `--device 0` | run YOLO on the CPU, or the first NVIDIA card (default: the card if there is one) |
| `--threads 1` | CPU threads for YOLO; more barely helps on a laptop but takes CPU from your sketch |
| `--smooth 0.5` | position smoothing; 0 = raw, closer to 1 = smoother but laggier |
| `--dark-objects` | track dark objects instead of people (for testing on a table) |

**On a laptop**, the tracker and a sketch compete for the CPU. If the sketch
stutters, try `--max-fps 15` (or lower) or `--imgsz 256`. With an NVIDIA
graphics card, the tracker can usually keep up with every frame of a fast
camera.

---

## 6. The OSC messages, and using them elsewhere

`track_osc.py` sends one UDP message per tracked person for every frame it
tracks (as often as the camera and computer allow, or `--max-fps`), to
`127.0.0.1:12000` by default:

```
/person  id x y
```

| Argument | Type | Meaning |
|---|---|---|
| `id` | int | the same while the person is tracked |
| `x` | float | metres from tape mark 1, along the width (0 to W) |
| `y` | float | metres from tape mark 1, along the height (0 to H) |

**Nothing is sent when someone leaves.** Forget anyone you haven't heard
about for half a second or so (that also covers lost packets). The tracker
sometimes loses someone for a moment and finds them again under a new id: the
Processing sketches' People tab re-attaches a new id that appears close to
someone just lost, so they keep their colour; do the same in your own code
if it matters.

Receiving them elsewhere:

- **Processing:** the `People` tab in `processing/template/` (no library
  needed). See [section 7](#7-make-your-own-processing-sketch).
- **TouchDesigner:** an **OSC In DAT** (or **OSC In CHOP**) on network port
  12000.
- **Max:** `[udpreceive 12000]` → `[route /person]`.
- **Pure Data:** `[netreceive -u -b 12000]` → `[oscparse]` →
  `[list trim]` → `[route person]`.
- **Python**, with `pip install python-osc`:

  ```python
  from pythonosc.dispatcher import Dispatcher
  from pythonosc.osc_server import BlockingOSCUDPServer

  def person(address, pid, x, y):
      print(f"person {pid} at {x:.2f}, {y:.2f} m")

  d = Dispatcher()
  d.map("/person", person)
  BlockingOSCUDPServer(("127.0.0.1", 12000), d).serve_forever()
  ```

To send to software on another computer, use `--osc-host` with its IP address,
and make sure its firewall lets UDP port 12000 in.

---

## 7. Make your own Processing sketch

1. Copy `processing/template` to `my-sketches/<name>/` (the `my-sketches/`
   folder is ignored by this repository, so your sketches stay yours: it can
   be a git repository of its own), and rename `template.pde` to
   `<name>.pde`.
2. Draw into `canvas`, which is the taped rectangle at `PIXELS_PER_METRE`
   (500) pixels per metre, origin at tape mark 1. The Projection tab warps it
   onto the floor.
3. `people` holds everyone on the floor: `p.x`, `p.y` in metres (or
   `p.canvasX()`, `p.canvasY()` in canvas pixels), smoothed between messages;
   `p.shown` fades from 0 to 1 as they arrive and back as they leave; `p.id`
   stays the same even when the tracker renumbers them.
4. `personArrived(p)` and `personLeft(p)` in your main tab are called as
   people come and go.

Keep the `People` and `Projection` tabs the same in every sketch (copy them
across if you change them). A sketch finds `calibration.json` in its own
folder or the nearest one above it, so it works anywhere inside this folder.

---

## 8. The files

| File | What it is |
|---|---|
| `calibrate.py` | step 4.1: camera ↔ projector ↔ floor calibration |
| `foot_calibrate.py` | step 4.2: correction for where feet land |
| `follow.py` | step 4.3: projects a circle under each person, to check tracking |
| `track_osc.py` | the tracker for running a piece: sends OSC, no projector window |
| `p_track.py` | a quick YOLO test on a camera or video, no calibration needed |
| `pipeline.py` | camera frame → people's feet → floor metres → tracks (shared by follow and track_osc) |
| `models.py` | loads YOLO for this machine's hardware (`--device`, `--threads`) |
| `tracking.py` | follows positions from frame to frame, smoothing them and keeping ids |
| `cameras.py` | frames from any camera or a video file; run it to list your cameras |
| `calib_utils.py` | the calibration maths, markers and window helpers |
| `osc.py` | sends the OSC messages (no extra package needed) |
| `recording.py` | records the camera view, projector or screen (keys c / p / f) |
| `calibration.json` | made by calibration: your space's measurements (not in the repository) |
| `processing/` | the Processing sketches |
| `requirements.txt`, `requirements-*.lock`, `lock_requirements.py` | the Python packages |
| `test_*.py` | checks for the calibration, tracking, OSC and recording code |

How they connect is drawn in the [guide](https://mattchu.github.io/yolo-people-2-the-floor/#files).

---

## 9. Troubleshooting

**Calibration finds few markers.**
- Make the room darker.
- Fix the exposure, or try another: see the end of 4.1.
- Use bigger markers: `--grid 6x4` or `--grid 5x3`.
- Aim the camera further down, so the projection fills more of its view.
  Markers seen at a shallow angle are too squashed to read.

**Feet are tracked in the wrong place.** Run `foot_calibrate.py` again. If
the camera or projector has moved, redo `calibrate.py` first.

**The sketch stutters while the tracker runs.** Lower the tracker's load:
`--max-fps 15`, `--imgsz 256`, keep `--threads 1`, or run the tracker on a
second computer with `--osc-host`.

**Nobody is detected.** People may be too small in the image: aim the camera
closer to the floor, or try `--imgsz 480`.

**A window lands on the wrong screen, or full screen misbehaves** (macOS in
particular). Use `--windowed` on `calibrate.py`, `foot_calibrate.py` and
`follow.py` to put the projector image in a normal window, then drag it to
the projector.

**Linux: Processing sketches fail with `UnsatisfiedLinkError`** (every
P2D/P3D sketch, even the examples). Close Processing, then add this line to
`~/.config/processing/preferences.txt`:

```
run.options=-Djdk.lang.Process.launchMechanism=VFORK
```

**Linux: a sketch using the Sound library dies with "Target VM failed to
initialize"** (PipeWire reporting 128 audio channels). Create `~/.asoundrc`
with the PipeWire default device, limited to 2 channels:

```
pcm.!default {
    type pipewire
    playback_node "-1"
    capture_node  "-1"
    channels 2
}
```

**Linux: windows look frozen after a full-screen window closes.** Turn your
desktop's compositor on (XFCE: Window Manager Tweaks → Compositor).

**Linux: screen recording (key f) doesn't work.** It needs an X11 session,
not Wayland.

**The camera disappears or can't be opened.** Check nothing else is using it
(only one program can at a time), and run `python cameras.py` again: numbers
can change when cameras are plugged in in a different order. A high-resolution
camera may need a USB 3 port and cable. Some laptops lose their USB 3
controller after sleep; a full shutdown, and on Macs an SMC reset, brings it
back.

---

## Licence

[MIT](LICENSE).
