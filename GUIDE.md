# ROS2 Line Follower — Ubuntu 22.04 + RTX 4080 Complete Guide

---

## Master Issues Table

Every problem found across all files. Check the Status column — all items marked **FIXED** have already been patched in the source code. Items marked **DOCUMENTED** are known limitations that do not need code changes. Items marked **IMPROVED** have been partially addressed with new functionality.

| # | Severity | Issue | Affected Files | Status |
|---|----------|-------|----------------|--------|
| 1 | CRITICAL | Hardcoded `/workspaces/ros2-line-follower/` path (11 locations) | `generate_dataset.py`, `train.py`, `evaluate.py`, `inspect_dataset.py`, `inference_node.py`, `overlay_recorder.py`, `trajectory_recorder.py` | **FIXED** |
| 2 | CRITICAL | PyTorch installs CPU-only build by default | `requirements.txt` | **FIXED** |
| 3 | CRITICAL | `from model import` fails unless CWD is `scripts/` | `train.py`, `evaluate.py` | **FIXED** |
| 4 | CRITICAL | `train.py` and `evaluate.py` use different random splits — evaluate contaminates results | `train.py` | **FIXED** |
| 5 | HIGH | `torch.load()` without `weights_only=True` — FutureWarning, will become error | `evaluate.py`, `inference_node.py`, `overlay_recorder.py` | **FIXED** |
| 6 | HIGH | `geometry_msgs` and `nav_msgs` not declared in `package.xml` | `package.xml` | **FIXED** |
| 7 | MEDIUM | Unused `ExecuteProcess` import causes flake8 F401 failure | `simulation.launch.py` | **FIXED** |
| 8 | MEDIUM | `ffmpeg` not in Docker image — needed for MP4 re-encode | `Dockerfile` | **FIXED** |
| 9 | LOW | `inference_node.py` always runs inference on CPU even with GPU available | `inference_node.py` | **DOCUMENTED** |
| 10 | LOW | Gazebo simulation forced headless with no way to view the 3D world | `simulation.launch.py`, `docker-compose.yml` | **IMPROVED** |
| 11 | LOW | `mp4v` codec may not be hardware-playable without re-encode | `overlay_recorder.py` | **DOCUMENTED** |
| 12 | LOW | Volume mount hides build artifacts — `colcon build` required on first container run | `Dockerfile`, `docker-compose.yml` | **DOCUMENTED** |
| 13 | LOW | `docker compose run` for every terminal creates isolated ROS2 domains — nodes cannot see each other's topics | `docker-compose.yml` | **FIXED** |

---

## What's New (Workstation-Setup Branch)

The following additions were made on top of the original code fixes to support running the project directly on the Ubuntu workstation. Every change is backward-compatible — the default headless workflow is unchanged.

| Addition | Description |
|----------|-------------|
| `docker exec -it` multi-terminal workflow | All terminals after the first must exec into the **same** container so ROS2 DDS discovery works. See Part E. |
| `gui:=true` launch argument | Enables the Gazebo 3D window on the Ubuntu desktop. Requires `xhost +` on the host once per session. |
| `world_name:=<name>` launch argument | Select any world file at launch time without editing source. |
| `rectangle.world` | Closed 6 × 4 m rectangular loop — robot drives continuously. |
| `corridor_maze.world` | 20 × 10 m arena with a 3-row snake line path, corridor walls, box obstacles, and cylinder pillars. |
| Foxglove live dashboard (Terminal D) | Stream all ROS2 topics to a browser in real time via `foxglove_bridge`. |
| X11 forwarding in `docker-compose.yml` | Host display and X11 socket mounted into the container so `gzclient` can open a window on the desktop. |

---

## Part A — Code Fixes Applied

Each fix below shows the exact file, the line numbers that changed, the original code, and the replacement.

---

### Fix 1 — Hardcoded paths in scripts

**Problem:** Every script hardcoded `/workspaces/ros2-line-follower/` — the GitHub Codespaces path — for all dataset and model file locations. On any other machine the paths do not exist and every script silently crashes with `FileNotFoundError`.

---

#### `src/line_follower/scripts/generate_dataset.py` — lines 1–11

**Before:**
```python
import os
import numpy as np
import cv2

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
SAMPLES_PER_CLASS = 1000
OUTPUT_DIR = '/workspaces/ros2-line-follower/dataset'
```

**After:**
```python
import os
import numpy as np
import cv2

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, '..', '..', '..'))

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
SAMPLES_PER_CLASS = 1000
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'dataset')
```

**Why it works:** `__file__` is always the script's own path. Three levels up from `scripts/` reaches the project root regardless of where you run the script from.

---

#### `src/line_follower/scripts/train.py` — lines 1–15

**Before:**
```python
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import LineFollowerCNN

DATASET_DIR = '/workspaces/ros2-line-follower/dataset'
MODEL_OUTPUT = '/workspaces/ros2-line-follower/line_follower_model.pth'
```

**After:**
```python
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, '..', '..', '..'))
sys.path.insert(0, _SCRIPTS_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import LineFollowerCNN

DATASET_DIR = os.path.join(_PROJECT_ROOT, 'dataset')
MODEL_OUTPUT = os.path.join(_PROJECT_ROOT, 'line_follower_model.pth')
```

---

#### `src/line_follower/scripts/evaluate.py` — lines 1–14

**Before:**
```python
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import LineFollowerCNN

DATASET_DIR = '/workspaces/ros2-line-follower/dataset'
MODEL_PATH = '/workspaces/ros2-line-follower/line_follower_model.pth'
```

**After:**
```python
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, '..', '..', '..'))
sys.path.insert(0, _SCRIPTS_DIR)

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import LineFollowerCNN

DATASET_DIR = os.path.join(_PROJECT_ROOT, 'dataset')
MODEL_PATH = os.path.join(_PROJECT_ROOT, 'line_follower_model.pth')
```

---

#### `src/line_follower/scripts/inspect_dataset.py` — lines 1–8 and line 21

**Before:**
```python
import os
import cv2
import numpy as np

DATASET = '/workspaces/ros2-line-follower/dataset'
...
cv2.imwrite('/workspaces/ros2-line-follower/dataset_preview.png', montage)
```

**After:**
```python
import os
import cv2
import numpy as np

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, '..', '..', '..'))

DATASET = os.path.join(_PROJECT_ROOT, 'dataset')
...
cv2.imwrite(os.path.join(_PROJECT_ROOT, 'dataset_preview.png'), montage)
```

---

### Fix 2 — Hardcoded paths in ROS2 nodes

**Problem:** ROS2 nodes are installed packages — `__file__` inside an installed node points to the `install/` directory, not the project root. A different strategy is needed: read the workspace path from an environment variable, with the original Codespaces path as fallback (so the project still works in Codespaces unchanged).

The environment variable `ROS2_LF_WORKSPACE` is set in both `Dockerfile` and `docker-compose.yml`.

---

#### `src/line_follower/line_follower/inference_node.py` — top of file and line 25–28

**Before:**
```python
import rclpy
from rclpy.node import Node
...
        self.model.load_state_dict(torch.load(
            '/workspaces/ros2-line-follower/line_follower_model.pth',
            map_location='cpu'
        ))
```

**After:**
```python
import os
import rclpy
from rclpy.node import Node
...
_WS = os.environ.get('ROS2_LF_WORKSPACE', '/workspaces/ros2-line-follower')
...
        self.model.load_state_dict(torch.load(
            os.path.join(_WS, 'line_follower_model.pth'),
            map_location='cpu',
            weights_only=True,
        ))
```

---

#### `src/line_follower/line_follower/overlay_recorder.py` — top of file, line 21–25, and line 92

**Before:**
```python
import rclpy
...
        self.model.load_state_dict(torch.load(
            '/workspaces/ros2-line-follower/line_follower_model.pth',
            map_location='cpu'))
...
        output_path = '/workspaces/ros2-line-follower/overlay_video.mp4'
```

**After:**
```python
import os
import rclpy
...
_WS = os.environ.get('ROS2_LF_WORKSPACE', '/workspaces/ros2-line-follower')
...
        self.model.load_state_dict(torch.load(
            os.path.join(_WS, 'line_follower_model.pth'),
            map_location='cpu',
            weights_only=True,
        ))
...
        output_path = os.path.join(_WS, 'overlay_video.mp4')
```

---

#### `src/line_follower/line_follower/trajectory_recorder.py` — top of file and line 55

**Before:**
```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
...
        output_path = '/workspaces/ros2-line-follower/trajectory.png'
```

**After:**
```python
import os
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
...
_WS = os.environ.get('ROS2_LF_WORKSPACE', '/workspaces/ros2-line-follower')
...
        output_path = os.path.join(_WS, 'trajectory.png')
```

---

### Fix 3 — PyTorch CUDA wheel

**Problem:** `requirements.txt` installed PyTorch from the default PyPI index, which gives the CPU-only build on Linux. The RTX 4080 was never used during training.

**File:** `requirements.txt`

**Before:**
```
torch==2.12.0
torchvision==0.27.0
```

**After:**
```
--extra-index-url https://download.pytorch.org/whl/cu121

torch==2.12.0+cu121
torchvision==0.27.0+cu121
```

**Why `--extra-index-url` instead of `--index-url`:** Using `--extra-index-url` keeps PyPI as the primary index for all other packages (numpy, opencv, etc.) and only adds the CUDA wheel server as a secondary source. `--index-url` would replace PyPI entirely and break all other package installs.

The `+cu121` suffix explicitly selects the CUDA 12.1 build, which supports the RTX 4080 (Ada Lovelace, compute capability 8.9, requires CUDA ≥ 11.8).

**If `torch==2.12.0+cu121` does not exist** on the wheel server when you build, the Dockerfile falls back to the latest available CUDA build automatically. You will see a pip error followed by a successful second install — this is expected.

---

### Fix 4 — Script import path (`from model import`)

**Problem:** `train.py` and `evaluate.py` used `from model import LineFollowerCNN` — a bare module import that only works when Python's current working directory is `scripts/`. Running either script from the project root or from inside a ROS2 node context caused `ModuleNotFoundError: No module named 'model'`.

**Already covered in Fix 1** — the `sys.path.insert(0, _SCRIPTS_DIR)` line added above each script's imports ensures `model.py` is always found by its directory location, independent of where you run the script from.

---

### Fix 5 — Train/evaluate random split mismatch

**Problem:** `evaluate.py` line 32 set `torch.manual_seed(42)` before calling `random_split`. `train.py` did not. This meant the 80/20 train/validation split used different random assignments in the two scripts. Images in the training set during `train.py` could appear in the validation set during `evaluate.py`, making accuracy metrics meaningless.

**File:** `src/line_follower/scripts/train.py`

**Before:**
```python
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
```

**After:**
```python
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
```

Now both scripts produce identical splits and evaluation results are valid.

---

### Fix 6 — `torch.load()` without `weights_only`

**Problem:** PyTorch 2.x prints a `FutureWarning` for every `torch.load()` call that omits `weights_only`. In future PyTorch versions this will become an error. Three files were affected.

State dictionaries are pure tensors and dictionaries, so `weights_only=True` is safe for all three.

**`src/line_follower/scripts/evaluate.py` — line 37:**

Before: `torch.load(MODEL_PATH)`

After: `torch.load(MODEL_PATH, map_location='cpu', weights_only=True)`

(`map_location='cpu'` also added — evaluate.py has no GPU setup, so explicit CPU load prevents a warning on GPU-less environments.)

**`src/line_follower/line_follower/inference_node.py` — lines 25–28:**

Before:
```python
torch.load(
    os.path.join(_WS, 'line_follower_model.pth'),
    map_location='cpu'
)
```

After:
```python
torch.load(
    os.path.join(_WS, 'line_follower_model.pth'),
    map_location='cpu',
    weights_only=True,
)
```

**`src/line_follower/line_follower/overlay_recorder.py` — lines 23–25:** Same change as inference_node.

---

### Fix 7 — Unused import in `simulation.launch.py`

**Problem:** `ExecuteProcess` was imported but never used in the launch file. This causes the `ament_flake8` CI check to fail with F401 (imported but unused).

**File:** `src/line_follower/launch/simulation.launch.py` — line 5

**Before:**
```python
from launch.actions import ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable
```

**After:**
```python
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
```

---

### Fix 8 — Missing `geometry_msgs` and `nav_msgs` in `package.xml`

**Problem:** `inference_node.py` publishes `geometry_msgs/Twist` and `trajectory_recorder.py` subscribes to `nav_msgs/Odometry`. Neither package was declared as a dependency. This means `rosdep install` would not install them, and strict ROS2 build environments (like CI with dependency resolution) would fail.

**File:** `src/line_follower/package.xml`

**Before:**
```xml
  <depend>rclpy</depend>
  <depend>sensor_msgs</depend>
  <depend>cv_bridge</depend>
```

**After:**
```xml
  <depend>rclpy</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>cv_bridge</depend>
```

---

### Fix 9 — `ffmpeg` missing from Docker image

**Problem:** The guide's troubleshooting section recommends re-encoding `overlay_video.mp4` with ffmpeg. Without ffmpeg in the Docker image, users would need to install it manually.

**File:** `Dockerfile`

**Before:**
```dockerfile
RUN apt-get update && apt-get install -y \
    ...
    xvfb \
    && rm -rf /var/lib/apt/lists/*
```

**After:**
```dockerfile
RUN apt-get update && apt-get install -y \
    ...
    xvfb \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
```

---

### Fix 10 — Separate containers break ROS2 DDS discovery

**Problem:** The original workflow opened every terminal with `docker compose run --rm ros2-line-follower bash`. Each invocation starts a **new, isolated container**. Even though all containers use `network_mode: host`, ROS2 DDS discovery only finds nodes within the same container process group by default with the `ROS_DOMAIN_ID` scoping. Nodes in separate containers could not reliably see each other's topics, causing the inference node to never receive camera frames.

**Fix:** Only **Terminal A** starts a new container with `docker compose run`. All subsequent terminals (B, C, D) use `docker exec -it` to open a new shell **inside the same already-running container**. See Part E for the exact commands.

---

## Part B — Documented Issues (No Code Change Required)

### Issue 9 — Inference always runs on CPU

`inference_node.py` loads the model with `map_location='cpu'` and never moves tensors to CUDA. The RTX 4080 is not used during live inference.

**Why it was not changed:** The CNN is tiny (~417k parameters, 48×64 inputs). On CPU it processes frames faster than the 30 Hz camera rate. Moving inference to GPU would require restructuring `image_callback` to move tensors, which adds complexity with no practical benefit for this application.

**If you want GPU inference** (for a larger model or higher frame rate), the change needed is:

```python
# In __init__:
self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
self.model = LineFollowerCNN(num_classes=3).to(self.device)
...
# In image_callback:
tensor = tensor.to(self.device)
```

---

### Issue 10 — Gazebo GUI (now available via `gui:=true`)

The simulation originally ran fully headless with no way to see the 3D world. This has been addressed — see Part E Step E8 for full instructions.

**How it now works:** `simulation.launch.py` accepts a `gui` launch argument (default: `false`). When set to `true`, `gzclient` opens a Gazebo 3D window on the Ubuntu desktop via X11. The window uses Mesa software rendering (`LIBGL_ALWAYS_SOFTWARE=1` remains set) so camera sensor rendering inside `gzserver` continues to work correctly. GPU-accelerated GUI rendering would require VirtualGL and is out of scope for this project.

---

### Issue 11 — `mp4v` codec and playback

`overlay_recorder.py` uses `cv2.VideoWriter_fourcc(*'mp4v')`. This creates a raw MPEG-4 Part 2 stream in an MP4 container. Some media players reject this. Re-encode after recording:

```bash
# Run inside the container — ffmpeg is now installed
ffmpeg -i /workspaces/ros2-line-follower/overlay_video.mp4 \
       -vcodec libx264 -crf 23 \
       /workspaces/ros2-line-follower/overlay_video_h264.mp4
```

---

### Issue 12 — Volume mount hides Docker build artifacts

The `Dockerfile` runs `colcon build` during the image build and writes `build/`, `install/`, `log/` into the image. When `docker compose run` mounts `.:/workspaces/ros2-line-follower`, the host directory (which does not yet contain `build/` or `install/`) overlays the image, hiding those directories.

**This means `colcon build` must be run inside the container every first session.** The workflow in Part E accounts for this.

After the first build inside the container, `build/`, `install/`, and `log/` appear on the host and persist across sessions. Subsequent container restarts skip the build step.

---

## Part C — Infrastructure Files

### `Dockerfile` (project root)

```dockerfile
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# Install ROS2 packages and system tools
RUN apt-get update && apt-get install -y \
    ros-humble-cv-bridge \
    ros-humble-xacro \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-robot-state-publisher \
    ros-humble-foxglove-bridge \
    python3-pip \
    python3-colcon-common-extensions \
    xvfb \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Must match the ROS2_LF_WORKSPACE env var and all node fallback paths
WORKDIR /workspaces/ros2-line-follower

# Install Python dependencies
# requirements.txt now points to the CUDA 12.1 wheel index for torch/torchvision
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt \
    || (echo "Pinned CUDA torch version not found — falling back to latest" && \
        pip3 install --no-cache-dir \
            numpy==1.26.4 \
            opencv-python-headless==4.10.0.84 \
            matplotlib==3.10.9 \
            pytest==9.1.0 && \
        pip3 install --no-cache-dir torch torchvision \
            --index-url https://download.pytorch.org/whl/cu121)

# Copy source into the image
COPY src/ src/

# Build the ROS2 package
RUN bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

# Source both setups automatically in every interactive bash shell
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /workspaces/ros2-line-follower/install/setup.bash" >> /root/.bashrc

ENV ROS_DOMAIN_ID=42
ENV ROS2_LF_WORKSPACE=/workspaces/ros2-line-follower
ENV LIBGL_ALWAYS_SOFTWARE=1
```

**Key decisions:**
- Base image `ros:humble-ros-base` is Ubuntu 22.04 + ROS2 Humble, the same as CI.
- `ros-humble-foxglove-bridge` is pre-installed so the Foxglove live dashboard (Terminal D) works without any extra setup.
- PyTorch CUDA wheels bundle their own CUDA runtime libraries. You do not need to install the CUDA toolkit inside the container — the host driver is injected by the NVIDIA Container Toolkit.
- `--symlink-install` in the build step means installed Python files are symlinks back to `src/`. After the volume mounts at runtime, edits to source files take effect immediately.

---

### `docker-compose.yml` (project root)

```yaml
services:
  ros2-line-follower:
    build: .
    image: ros2-line-follower:latest

    volumes:
      # Mount at exactly this path — matches ROS2_LF_WORKSPACE and all node fallback paths
      - .:/workspaces/ros2-line-follower
      # X11 socket — required for Gazebo GUI (gui:=true). Safe to keep even when running headless.
      - /tmp/.X11-unix:/tmp/.X11-unix

    environment:
      - LIBGL_ALWAYS_SOFTWARE=1
      - ROS_DOMAIN_ID=42
      - ROS2_LF_WORKSPACE=/workspaces/ros2-line-follower
      # Pass host display so gzclient can open a window. Empty when running headless (gui:=false).
      - DISPLAY=${DISPLAY}

    # GPU passthrough — requires NVIDIA Container Toolkit on the host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

    # Host networking keeps ROS2 DDS discovery working across terminals
    network_mode: host

    stdin_open: true
    tty: true
```

**Key decisions:**
- `DISPLAY=${DISPLAY}` substitutes the host's X11 display variable (e.g. `:1`) at container start. When `$DISPLAY` is unset on the host (SSH without X11 forwarding), the variable is empty and `simulation.launch.py` falls back to starting its own Xvfb virtual display — so headless mode is unaffected.
- `/tmp/.X11-unix:/tmp/.X11-unix` mounts the host's X11 socket directory into the container. This is the channel through which `gzclient` draws its window on the Ubuntu desktop. The mount is harmless when running headless.
- `network_mode: host` is required for ROS2 DDS topic discovery to work between multiple container shells opened simultaneously. Without this, the simulation terminal and inference terminal cannot see each other's topics.
- Volume mount path MUST be `/workspaces/ros2-line-follower` — all ROS2 node fallback paths and the `ROS2_LF_WORKSPACE` value depend on this exact string.

---

### `src/line_follower/launch/simulation.launch.py` (refactored)

The launch file was refactored to support runtime configuration via launch arguments. The `OpaqueFunction` pattern is used so arguments can be resolved to Python strings before constructing file paths.

**Launch arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `world_name` | `line_track` | Name of the world file (without `.world`) from the `worlds/` directory |
| `gui` | `false` | Set to `true` to open a Gazebo 3D window on the host desktop |

**Usage examples:**
```bash
# Default headless run
ros2 launch line_follower simulation.launch.py

# Specific world, headless
ros2 launch line_follower simulation.launch.py world_name:=rectangle

# Specific world with Gazebo window visible
ros2 launch line_follower simulation.launch.py world_name:=corridor_maze gui:=true
```

**How `_ensure_xvfb()` now behaves:**

```
DISPLAY set in container?
├─ YES (host passed DISPLAY=:1) → _ensure_xvfb() returns immediately
│   gzserver renders on the host X server (with LIBGL_ALWAYS_SOFTWARE=1)
│   gui:=true → gzclient opens a window on the Ubuntu desktop
└─ NO (headless run, DISPLAY empty) → _ensure_xvfb() starts Xvfb on :99
    gzserver renders into the virtual display
    gui:=true in this state will fail — do not combine headless with gui
```

---

### World files

Three world files are available. All dark line segments are thin boxes at `z=0.001` so they sit just above the ground plane. The ground plane is light; the boxes are nearly black — the CNN sees this contrast.

The robot always spawns at **(0, 0) facing +X**. All worlds place the start of the line path at or through this point.

#### `src/line_follower/worlds/line_track.world` (default)

Simple two-segment open path: one straight section followed by a gentle diagonal. The original world shipped with the project.

#### `src/line_follower/worlds/rectangle.world`

Closed 6 × 4 m rectangular loop. The robot starts at (0, 0) on the bottom segment and drives continuously around the loop. Vertical segments are 0.1 m longer than the rectangle height so their ends slightly overlap the horizontal segments at each corner, closing the gap.

```
(-3,4) ─────── top ──────── (3,4)
  │                            │
 left                        right
  │                            │
(-3,0) ────── bottom ─────── (3,0)
                  * robot
```

#### `src/line_follower/worlds/corridor_maze.world`

20 × 10 m enclosed arena with a 3-row snake line path (54 m total line length). A 3 mm dark line winds through two corridor divider walls, four wall stubs, four box obstacles (burnt orange), and three cylinder pillars (dark red). All obstacles have collision geometry so the robot physically interacts with them.

```
(-9,4) ═══════════════ Row 3 ═══════════════ (9,4)
║ left turn at x=-9
(-9,2) ═══════════════ Row 2 ═══════════════ (9,2)
                               right turn at x=9 ║
(-9,0) ═══════════════ Row 1 ═══════════════ (9,0)
                  * robot starts here
```

Outer walls (gray) at x=±10, y=±5 physically bound the arena. Corridor divider walls (blue-gray) at y=1 and y=3 create the three corridors, with gaps left at the turn points.

**CNN behaviour in the maze:** The model was trained on synthetic straight-line images. It handles the gentle 90° corners (via its search-and-rotate recovery behaviour) but may need several seconds to re-acquire the line at each turn.

---

## Part D — Host Machine Setup (Ubuntu 22.04)

Complete these steps on the Ubuntu host before touching the project.

### Step D1 — Verify Ubuntu version

```bash
lsb_release -a
# Must show: Ubuntu 22.04.x LTS
```

### Step D2 — Install NVIDIA drivers

```bash
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
```

After reboot, verify:

```bash
nvidia-smi
# Must show RTX 4080 with driver version ≥ 525 and CUDA version
```

If `nvidia-smi` fails, stop here. Do not proceed without working drivers.

### Step D3 — Install Docker Engine

Do not use `docker.io` from Ubuntu's default apt repos — it is outdated.

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Allow running docker without sudo
sudo usermod -aG docker $USER
newgrp docker
```

Verify:

```bash
docker run --rm hello-world
docker compose version
# Must show Compose version v2.x
```

### Step D4 — Install NVIDIA Container Toolkit

This is what allows `--gpus all` to actually pass through the GPU.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU is visible inside containers:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
# Must show RTX 4080 inside the container output
```

If this command fails, fix it before continuing. Everything else depends on this working.

### Step D5 — Allow X11 connections (required for Gazebo GUI only)

This step is only needed if you want to see the Gazebo 3D window on the desktop (`gui:=true`). It resets on every reboot.

```bash
# Run on the host, outside Docker, once per session
xhost +
```

This disables X11 access control, allowing the Docker container (which runs as root) to draw windows on the host display. To re-enable access control after you are done:

```bash
xhost -
```

> **Note:** `xhost +` allows any local process to connect to the X server. On a shared workstation, use `xhost +local:` instead, which restricts connections to the local machine. The container runs as root, so `xhost +local:root` is also valid.

---

## Part E — Build and Run Workflow

### Step E1 — Build the Docker image

Navigate to the project root (where `Dockerfile` lives) and run:

```bash
cd ~/ros2-line-follower

docker compose build
```

Expected output (abbreviated):
```
[+] Building ...
 => [1/7] FROM docker.io/library/ros:humble-ros-base
 => [2/7] RUN apt-get update && apt-get install -y ...
 => [3/7] WORKDIR /workspaces/ros2-line-follower
 => [4/7] COPY requirements.txt .
 => [5/7] RUN pip3 install --no-cache-dir -r requirements.txt ...
 => [6/7] COPY src/ src/
 => [7/7] RUN bash -c "source /opt/ros/humble/setup.bash && colcon build ..."
 => exporting to image
```

The PyTorch CUDA download is the largest step (~2.5 GB) and takes a few minutes.

---

### Step E2 — One-time setup inside the container

Open a container shell (this becomes Terminal A for the simulation later):

```bash
docker compose run --rm ros2-line-follower bash
```

Inside the container, source ROS2 and build the package:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Successful output ends with:
```
Summary: 1 package finished [...]
```

This only needs to be done once — after the first build, `build/`, `install/`, and `log/` exist on the host and persist across container restarts.

---

### Step E3 — Verify GPU is available

```bash
# Inside container
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')
print('PyTorch version:', torch.__version__)
"
```

Expected:
```
CUDA available: True
Device: NVIDIA GeForce RTX 4080
PyTorch version: 2.12.0+cu121
```

If `CUDA available: False`, stop and see Troubleshooting.

---

### Step E4 — Generate training dataset

```bash
# Inside container — can run from any directory
python3 /workspaces/ros2-line-follower/src/line_follower/scripts/generate_dataset.py
```

Expected:
```
Generated 1000 images for class "left"
Generated 1000 images for class "center"
Generated 1000 images for class "right"
Dataset complete at /workspaces/ros2-line-follower/dataset
```

On the host machine, `dataset/` appears in the project root with ~200 MB of PNG files.

---

### Step E5 — Train the model

```bash
# Inside container — can run from any directory
python3 /workspaces/ros2-line-follower/src/line_follower/scripts/train.py
```

The first two lines confirm what device is used:

```
Classes found: ['center', 'left', 'right']
Total images: 3000
Training: 2400, Validation: 600
Training on: cuda          ← must say cuda, not cpu
```

Full expected output (12 epochs, under 2 minutes on RTX 4080):
```
Epoch  1/12 | Train loss: 0.8234 | Train acc: 62.71% | Val acc: 71.33%
Epoch  2/12 | Train loss: 0.3415 | Train acc: 87.42% | Val acc: 89.17%
...
Epoch 12/12 | Train loss: 0.0412 | Train acc: 98.75% | Val acc: 97.83%
Model saved to /workspaces/ros2-line-follower/line_follower_model.pth
```

`line_follower_model.pth` (~1.6 MB) appears in the project root on the host.

---

### Step E6 — Evaluate the model (optional)

```bash
# Inside container
python3 /workspaces/ros2-line-follower/src/line_follower/scripts/evaluate.py
```

Expected output includes overall accuracy and confusion matrix. Accuracy above 95% is expected with the synthetic dataset.

---

### Understanding the multi-terminal setup

Running the simulation requires up to four terminals open simultaneously, each running a different process. All four **must be inside the same container** — if you open new containers for each terminal, their ROS2 nodes run in isolation and cannot communicate.

**The correct pattern:**

```
Host terminal 1 → docker compose run → [Container A] → Terminal A (simulation)
Host terminal 2 → docker exec -it   → [Container A] → Terminal B (inference)
Host terminal 3 → docker exec -it   → [Container A] → Terminal C (recorders)
Host terminal 4 → docker exec -it   → [Container A] → Terminal D (Foxglove)
```

To find the running container name after Terminal A starts:
```bash
# On the host, in any terminal:
docker ps --format "table {{.Names}}\t{{.Status}}"
# Look for the ros2-line-follower container — name looks like:
# ros2-line-follower-ros2-line-follower-run-<id>
```

**Why `docker exec` requires `-it`:** Without `-i` (keep stdin open) and `-t` (allocate a pseudo-TTY), bash has no terminal to attach to and exits immediately. You are silently dropped back to the host shell. The container keeps running, but your commands execute on the host — which has no ROS2 and will fail with `ros2: command not found` or `Package not found`.

**Why sourcing is required in every exec session:** Each `docker exec` bash session starts clean. The `/root/.bashrc` that sources setup files only runs in interactive login shells started by the container's entrypoint. `docker exec` bypasses the entrypoint, so you must source manually.

---

### Step E7 — Terminal A: Launch the Gazebo simulation

Open a host terminal and start the container:

```bash
# [host]
docker compose run --rm ros2-line-follower bash
```

If you want the Gazebo window visible on the desktop, run this on the host first (once per session):
```bash
xhost +
```

Inside the container, source and launch:

```bash
# [container — Terminal A]
source /opt/ros/humble/setup.bash
source /workspaces/ros2-line-follower/install/setup.bash

# Headless (default):
ros2 launch line_follower simulation.launch.py

# With Gazebo window on the desktop:
ros2 launch line_follower simulation.launch.py gui:=true

# Different world:
ros2 launch line_follower simulation.launch.py world_name:=rectangle

# Different world with GUI:
ros2 launch line_follower simulation.launch.py world_name:=corridor_maze gui:=true
```

Wait until you see all of the following before proceeding:

```
[gzserver-1] Gazebo multi-robot simulator, version 11.x
[spawn_entity.py-3] [INFO] ... Spawn status: SpawnEntity: Successfully spawned entity [line_follower]
[camera_node-4] [INFO] ... Camera node started, waiting for images...
[camera_node-4] [INFO] ... Received image: 640x480, encoding: rgb8
```

The last line confirms the camera is publishing. On this workstation it appears roughly 2 seconds after the robot spawns. Only proceed to Terminal B after seeing it.

---

### Step E8 — Terminal B: AI inference node

Open a **new host terminal** (do not open a new container):

```bash
# [host]
docker exec -it ros2-line-follower-ros2-line-follower-run-<id> bash
```

Replace `<id>` with the container name from `docker ps`.

```bash
# [container — Terminal B]
source /opt/ros/humble/setup.bash
source /workspaces/ros2-line-follower/install/setup.bash
ros2 run line_follower inference_node
```

Expected continuous output:
```
[inference_node] [INFO] ... Inference node started, model loaded
[inference_node] [INFO] ... Prediction: center (conf 0.94) → cmd_vel
[inference_node] [INFO] ... Prediction: center (conf 0.91) → cmd_vel
[inference_node] [INFO] ... Prediction: left (conf 0.87) → cmd_vel
```

The robot is now driving in the simulation.

---

### Step E9 — Terminal C: Record trajectory and/or annotated video

Open a **new host terminal**:

```bash
# [host]
docker exec -it ros2-line-follower-ros2-line-follower-run-<id> bash
```

```bash
# [container — Terminal C]
source /opt/ros/humble/setup.bash
source /workspaces/ros2-line-follower/install/setup.bash

# Record trajectory (saves on Ctrl+C):
ros2 run line_follower trajectory_recorder

# OR record annotated video (saves on Ctrl+C):
ros2 run line_follower overlay_recorder
```

Let either recorder run for at least 30 seconds, then press `Ctrl+C`.

Output files in the project root on the host:
- `trajectory_recorder` → `trajectory.png`
- `overlay_recorder` → `overlay_video.mp4`

**Stop order matters** — if you stop things in the wrong order the video file will not be written:
1. `Ctrl+C` in Terminal C first — wait for the "Saved X frames" line to appear
2. `Ctrl+C` in Terminal B (inference)
3. `Ctrl+C` in Terminal A (simulation)

If `overlay_video.mp4` is unplayable, re-encode:

```bash
ffmpeg -i /workspaces/ros2-line-follower/overlay_video.mp4 \
       -vcodec libx264 -crf 23 \
       /workspaces/ros2-line-follower/overlay_video_h264.mp4
```

---

### Step E10 — Terminal D: Live Foxglove dashboard (optional)

Foxglove Studio is a browser-based ROS2 visualiser. It connects via WebSocket to `foxglove_bridge` running inside the container and shows the camera feed, plots, odometry, and robot model in real time — no recording needed.

Open a **new host terminal**:

```bash
# [host]
docker exec -it ros2-line-follower-ros2-line-follower-run-<id> bash
```

```bash
# [container — Terminal D]
source /opt/ros/humble/setup.bash
source /workspaces/ros2-line-follower/install/setup.bash
ros2 run foxglove_bridge foxglove_bridge
```

Expected on start:
```
[INFO] [foxglove_bridge]: Starting Foxglove bridge on port 8765
```

**To connect — two options:**

**Option 1 — browser on this machine:**
1. Open `https://app.foxglove.dev` in a browser.
2. Click **Open connection** → **Foxglove WebSocket**.
3. Enter `ws://localhost:8765` → click **Open**.

**Option 2 — from another machine on the same network:**
1. Find the workstation's LAN IP: `ip route get 1 | awk '{print $7; exit}'`
2. Open Foxglove on the other machine, connect to `ws://<workstation-ip>:8765`.

No port forwarding is needed because `network_mode: host` exposes port 8765 directly on the host's network interfaces.

**Useful panels to add in Foxglove:**

| Panel | Topic | What you see |
|-------|-------|-------------|
| Image | `/camera/image_raw` | Live camera feed from the robot |
| Plot | `/cmd_vel` → `linear.x`, `angular.z` | Steering commands over time |
| Odometry / 3D | `/odom` | Robot position in the world |
| Robot Model | URDF | Robot joints and body in 3D |

Terminal D is independent — stop it with `Ctrl+C` at any time with no side effects on the simulation.

---

### Step E11 — Switching between worlds

No rebuild is needed to switch worlds — world files are symlinked by `--symlink-install`. Simply stop Terminal A and relaunch with a different `world_name`:

```bash
# [container — Terminal A, after stopping previous run]
ros2 launch line_follower simulation.launch.py world_name:=rectangle
ros2 launch line_follower simulation.launch.py world_name:=corridor_maze
ros2 launch line_follower simulation.launch.py world_name:=line_track   # back to default
```

Available worlds:

| `world_name` | Shape | Notes |
|---|---|---|
| `line_track` | Short open path with one bend | Default |
| `rectangle` | Closed 6 × 4 m rectangular loop | Robot drives continuously |
| `corridor_maze` | 3-row snake in 20 × 10 m arena | Walls and obstacles; 54 m total line |

---

### Step E12 — Inspect dataset preview (optional)

```bash
# Inside container — any terminal
python3 /workspaces/ros2-line-follower/src/line_follower/scripts/inspect_dataset.py
```

Output file: `dataset_preview.png` in the project root. Shows a 3×5 montage (3 classes × 5 samples).

---

## Part F — Verification Commands

Run these at any point to confirm the environment is healthy:

```bash
# ROS2 package is discoverable
ros2 pkg list | grep line_follower

# All four executables are registered
ros2 pkg executables line_follower
# Expected:
#   line_follower camera_node
#   line_follower inference_node
#   line_follower overlay_recorder
#   line_follower trajectory_recorder

# GPU is reachable from PyTorch
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Model file exists after training
ls -lh /workspaces/ros2-line-follower/line_follower_model.pth
# Expected: around 1.6 MB

# Dataset exists after generation
ls /workspaces/ros2-line-follower/dataset/
# Expected: center/  left/  right/

# All three world files are installed
ls $(ros2 pkg prefix line_follower)/share/line_follower/worlds/
# Expected: corridor_maze.world  line_track.world  rectangle.world

# Unit tests pass
cd /workspaces/ros2-line-follower
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/line_follower/test/test_model_smoke.py -v
# Expected: 3 tests passed
```

---

## Part G — Troubleshooting

### `torch.cuda.is_available()` returns `False`

**Check 1 — Which PyTorch was installed?**
```bash
python3 -c "import torch; print(torch.__version__)"
```
If the version ends in `+cpu`, the CUDA build was not installed. Reinstall:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

**Check 2 — Is the GPU visible inside Docker?**
```bash
nvidia-smi
```
If this fails inside the container, the NVIDIA Container Toolkit on the host is not configured. Run on the host:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
Then rebuild: `docker compose build`.

**Check 3 — Was the container started with GPU access?**
Confirm you used `docker compose run`, not `docker run`. The `docker-compose.yml` includes the GPU reservation; a plain `docker run` without `--gpus all` will not pass through the GPU.

---

### `Training on: cpu` even though CUDA is available

`train.py` prints the device on the third output line. If it says `cpu` despite `torch.cuda.is_available()` returning `True`, there is a logic error. This should not happen with the current code. Double-check you have run `colcon build` and sourced `install/setup.bash` after any source changes.

---

### `FileNotFoundError: .../line_follower_model.pth`

The model file does not exist yet. You must complete Step E5 (training) before Step E8 (inference). Verify:
```bash
ls /workspaces/ros2-line-follower/line_follower_model.pth
```
If missing, run `python3 /workspaces/ros2-line-follower/src/line_follower/scripts/train.py`.

---

### `ModuleNotFoundError: No module named 'model'`

After the path fixes applied in this guide, this should not occur. If it does, the `sys.path.insert` line is missing. Verify the top of `train.py` or `evaluate.py` starts with:
```python
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)
```

---

### `ros2: command not found` or `Package 'line_follower' not found`

**Most common cause:** You are in a `docker exec` shell that has not been sourced, or you ran a `ros2` command on the host by accident (see the `-it` warning below).

Every new shell — whether from `docker compose run` or `docker exec` — needs:
```bash
source /opt/ros/humble/setup.bash
source /workspaces/ros2-line-follower/install/setup.bash
```

---

### `docker exec` drops back to host shell immediately

You forgot the `-it` flags. Without them, bash starts and exits instantly because it has no terminal to attach to. Your next command runs on the **host** machine, not in the container.

```bash
# Wrong — exits immediately, drops to host
docker exec ros2-line-follower-ros2-line-follower-run-<id> bash

# Correct
docker exec -it ros2-line-follower-ros2-line-follower-run-<id> bash
```

---

### Simulation fails: `[gzserver] process has died`

**Check 1 — Are all Gazebo packages installed?**
```bash
dpkg -l | grep ros-humble-gazebo
# Must list: ros-humble-gazebo-ros-pkgs
```
If missing, the Docker build was not run correctly. Rebuild: `docker compose build`.

**Check 2 — Is DISPLAY set?**
```bash
echo $DISPLAY
```
If empty and you are running headless, check that `_ensure_xvfb()` is not failing silently. Set manually and retry:
```bash
Xvfb :99 -screen 0 1280x720x24 &
export DISPLAY=:99
ros2 launch line_follower simulation.launch.py
```

---

### Gazebo window does not appear when using `gui:=true`

**Check 1 — Did you run `xhost +` on the host?**
```bash
xhost +
```
Run this on the host (not inside the container), then restart the simulation.

**Check 2 — Is the container newly started?**
The `/tmp/.X11-unix` volume mount and `DISPLAY=${DISPLAY}` only take effect when a container is **started**, not exec'd into. If you are in a container that was started before these were added to `docker-compose.yml`, stop it and start a new one with `docker compose run`.

**Check 3 — Is `$DISPLAY` set in the host shell?**
```bash
echo $DISPLAY
# Should print :1 (or similar)
```
If empty (you are in an SSH session without X11 forwarding), GUI cannot work. Use the Foxglove dashboard (Terminal D) instead for remote monitoring.

---

### Inference node starts but robot does not move

**Check 1 — Is the simulation running in another terminal?**
The inference node publishes to `/cmd_vel`. Without a simulation, there is nothing listening.

**Check 2 — Are topics visible?**
```bash
ros2 topic list
# Must include: /camera/image_raw  /cmd_vel  /odom
```
If `/camera/image_raw` is missing, the simulation is not running or the camera plugin failed to load. Check Terminal A output.

**Check 3 — Are both terminals in the same container?**
```bash
# In Terminal B, check the container hostname:
hostname
# In Terminal A, check the container hostname:
hostname
# Both must print the same value
```
If they differ, one terminal is in a separate container. Stop both, restart Terminal A with `docker compose run`, then open Terminal B with `docker exec -it`.

**Check 4 — Is the confidence threshold being hit?**
If all predictions show `conf < 0.8`, the robot pauses then spins searching. This means the CNN is not recognising the line. Check that `train.py` completed successfully and the model file is from a recent training run (accuracy > 95%).

---

### `overlay_video.mp4` is created but unplayable

Re-encode with H.264 (ffmpeg is pre-installed in the container):
```bash
ffmpeg -i /workspaces/ros2-line-follower/overlay_video.mp4 \
       -vcodec libx264 -crf 23 \
       /workspaces/ros2-line-follower/overlay_video_h264.mp4
```

---

## Part H — Complete File Reference After All Changes

### Files modified from original

| File | What Changed |
|------|-------------|
| `src/line_follower/scripts/generate_dataset.py` | `OUTPUT_DIR` computed from `__file__` |
| `src/line_follower/scripts/train.py` | Path from `__file__`, `sys.path.insert`, `torch.manual_seed(42)` |
| `src/line_follower/scripts/evaluate.py` | Path from `__file__`, `sys.path.insert`, `weights_only=True`, `map_location='cpu'` |
| `src/line_follower/scripts/inspect_dataset.py` | All paths computed from `__file__` |
| `src/line_follower/line_follower/inference_node.py` | `os` import, `_WS` env var, `weights_only=True` |
| `src/line_follower/line_follower/overlay_recorder.py` | `os` import, `_WS` env var, `weights_only=True` |
| `src/line_follower/line_follower/trajectory_recorder.py` | `os` import, `_WS` env var |
| `src/line_follower/launch/simulation.launch.py` | Removed unused import; refactored with `OpaqueFunction`; added `world_name` and `gui` launch arguments |
| `src/line_follower/package.xml` | Added `geometry_msgs`, `nav_msgs` dependencies |
| `src/line_follower/setup.py` | Added `rectangle.world` and `corridor_maze.world` to `data_files` |
| `requirements.txt` | Added `--extra-index-url`, `+cu121` suffixes on torch/torchvision |
| `docker-compose.yml` | Added `DISPLAY=${DISPLAY}` env var and `/tmp/.X11-unix` volume for Gazebo GUI |
| `.gitignore` | Added `test-folder/` |

### Files created

| File | Purpose |
|------|---------|
| `Dockerfile` | Builds Ubuntu 22.04 + ROS2 Humble + CUDA PyTorch image |
| `docker-compose.yml` | Mounts workspace, passes GPU, X11 socket, and `ROS2_LF_WORKSPACE` |
| `src/line_follower/worlds/rectangle.world` | Closed 6 × 4 m rectangular loop |
| `src/line_follower/worlds/corridor_maze.world` | 3-row snake path in 20 × 10 m arena with walls and obstacles |

### Files unchanged

| File | Reason |
|------|--------|
| `src/line_follower/line_follower/camera_node.py` | No hardcoded paths, no imports to fix |
| `src/line_follower/line_follower/model.py` | Pure PyTorch, no paths |
| `src/line_follower/scripts/model.py` | Same |
| `src/line_follower/urdf/robot.urdf.xacro` | XML, no paths |
| `src/line_follower/worlds/line_track.world` | Original default world, unchanged |
| `src/line_follower/test/test_model_smoke.py` | Already had correct `sys.path.insert` |
| `.github/workflows/ci.yml` | CI installs CPU torch, acceptable for tests |
| `.devcontainer/devcontainer.json` | Codespaces only, not used in Docker workflow |

---

## Make Commands

All common workflows are wrapped as `make` targets. Run `make help` to see the list at any time.

### Launcher

| Command | Description |
|---------|-------------|
| `make gui` | Launch the graphical launcher (recommended) |

### One-time Setup

| Command | Description |
|---------|-------------|
| `make build` | Build the Docker image |
| `make colcon-build` | Build the ROS2 package inside the container |

### Dataset & Training *(no simulation needed)*

| Command | Description |
|---------|-------------|
| `make dataset` | Generate training dataset (~3000 images) |
| `make train` | Train the CNN model |
| `make evaluate` | Evaluate model accuracy (optional) |
| `make inspect` | Save dataset preview image (optional) |

### Simulation *(run in Terminal A — blocks until Ctrl+C)*

| Command | Description |
|---------|-------------|
| `make sim` | Headless, default world (`line_track`) |
| `make sim-gui` | With Gazebo window, default world |
| `make sim-rect` | Headless, rectangle world |
| `make sim-maze` | Headless, corridor_maze world |

### Nodes *(run in separate terminals while simulation is running)*

| Command | Description |
|---------|-------------|
| `make inference` | Run the AI inference node |
| `make record-traj` | Record trajectory → `trajectory.png` |
| `make record-video` | Record annotated video → `overlay_video.mp4` |
| `make foxglove` | Foxglove bridge at `ws://localhost:8765` |

### Stop

| Command | Description |
|---------|-------------|
| `make stop` | Stop all containers (`docker compose down`) |

---

## Quick Command Reference

```bash
# ── One-time: build image ──────────────────────────────────────────────────
docker compose build

# ── Every session: open Terminal A ────────────────────────────────────────
docker compose run --rm ros2-line-follower bash

# ── Every first session: build ROS2 package ───────────────────────────────
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# ── Open Terminals B / C / D (exec into same container as A) ─────────────
docker exec -it ros2-line-follower-ros2-line-follower-run-<id> bash
# Then in each exec shell:
source /opt/ros/humble/setup.bash
source /workspaces/ros2-line-follower/install/setup.bash

# ── Find the running container name ───────────────────────────────────────
docker ps --format "table {{.Names}}\t{{.Status}}"

# ── Verify GPU ─────────────────────────────────────────────────────────────
python3 -c "import torch; print(torch.cuda.is_available())"

# ── Generate dataset (one-time) ────────────────────────────────────────────
python3 /workspaces/ros2-line-follower/src/line_follower/scripts/generate_dataset.py

# ── Train model (one-time or when improving) ───────────────────────────────
python3 /workspaces/ros2-line-follower/src/line_follower/scripts/train.py

# ── Evaluate model (optional) ──────────────────────────────────────────────
python3 /workspaces/ros2-line-follower/src/line_follower/scripts/evaluate.py

# ── Terminal A: simulation (headless default) ──────────────────────────────
ros2 launch line_follower simulation.launch.py

# ── Terminal A: simulation with Gazebo window (run xhost + on host first) ──
xhost +   # on host before starting the container
ros2 launch line_follower simulation.launch.py gui:=true

# ── Terminal A: switch world ───────────────────────────────────────────────
ros2 launch line_follower simulation.launch.py world_name:=rectangle
ros2 launch line_follower simulation.launch.py world_name:=corridor_maze
ros2 launch line_follower simulation.launch.py world_name:=corridor_maze gui:=true

# ── Terminal B: inference node ─────────────────────────────────────────────
ros2 run line_follower inference_node

# ── Terminal C: trajectory recorder (Ctrl+C to save) ──────────────────────
ros2 run line_follower trajectory_recorder

# ── Terminal C: video recorder (Ctrl+C to save) ────────────────────────────
ros2 run line_follower overlay_recorder

# ── Re-encode video if unplayable ──────────────────────────────────────────
ffmpeg -i overlay_video.mp4 -vcodec libx264 -crf 23 overlay_video_h264.mp4

# ── Terminal D: Foxglove live dashboard ────────────────────────────────────
ros2 run foxglove_bridge foxglove_bridge
# Then open https://app.foxglove.dev → Open connection → ws://localhost:8765
```
