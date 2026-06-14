# 🤖 ROS2 Line-Following Robot — Vision-Based, Trained in the Cloud

A complete, reproducible robotics + AI project: a differential-drive robot that follows a line using a convolutional neural network for vision — built, trained, and simulated **entirely inside a GitHub Codespace**, with no GPU and no local install required.

The project is organised into phases, each building on the last — follow them in order to go from an empty repo to a working line-following robot.

---

## What this project demonstrates

- A fully reproducible ROS2 + PyTorch development environment defined in a single dev container — clone, open in Codespaces, and everything works.
- A simulated robot (URDF) with a camera, driven in headless Gazebo with no display or GPU.
- A convolutional neural network trained on auto-labelled, augmented synthetic data to classify where a line is (left / center / right).
- A closed perception-to-action loop: camera frames → CNN → steering commands, with a confidence-based recovery behaviour when the line is lost.
- Three ways to **see** the results despite running headless: a trajectory plot, an annotated camera video, and a live Foxglove dashboard.

---

## Make this your own (start here)

If you want to follow along, experiment, and **save your own work**, the recommended first step is to **fork** this repository.

**What's a fork?** A fork is your own personal copy of this repo under your own GitHub account. You own it completely — you can change anything, and your changes never affect the original. It's the standard way to take someone else's project and build on it.

**Why fork before doing anything else?** You can open *this* repo directly in a Codespace and edit it, but you won't be able to save (`git push`) your changes back, because you don't own it. A fork gives you a copy you *can* push to — so your progress is saved to GitHub, not just living in a temporary Codespace.

**How to do it:**

1. Click the **Fork** button at the top-right of this repository's page. GitHub creates `your-username/ros2-line-follower` under your account.
2. On **your fork's** page, click the green **Code** button → **Codespaces** tab → **Create codespace on main**.
3. Develop freely. Commit and push as you go — your work saves to your own fork.

> **Fork vs Clone vs Branch — the quick version:**
> - **Fork** — a copy under *your* GitHub account. Use this to make the project your own.
> - **Clone** — downloading a repo onto a computer to work locally. You'd typically clone *your fork*.
> - **Branch** — a separate line of work *inside* a repo, for trying things without disturbing your main copy. Handy once you're comfortable, but not needed to get started.

If you only want to *try it* without saving changes, you can skip forking and use the one-click Codespace below — just know your edits won't be saved to GitHub.

---

## Quick start (one-click Codespace)

The fastest path — no local setup at all:

1. Click the green **Code** button on this repository → **Codespaces** tab → **Create codespace on main**.
2. Wait for the container to build (a few minutes the first time — it installs ROS2 packages and Python dependencies automatically).
3. Open a terminal and run the simulation (see [Running it](#running-it) below).

Everything needed — ROS2 Humble, PyTorch, Gazebo, the Foxglove bridge — is installed automatically by the dev container definition in [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json).

### Running it locally instead

If you prefer to run on your own machine, you'll need: Ubuntu 22.04, ROS2 Humble, and Python 3.10. Clone the repo, then reproduce the dev container's setup steps manually:

```bash
sudo apt-get update && sudo apt-get install -y \
  python3-pip python3-colcon-common-extensions \
  ros-humble-cv-bridge ros-humble-gazebo-ros-pkgs \
  ros-humble-robot-state-publisher ros-humble-xacro \
  ros-humble-foxglove-bridge xvfb nano

pip3 install 'numpy<2' torch torchvision opencv-python-headless matplotlib jupyterlab

echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

(On a machine with a real display and GPU, Gazebo will run faster and show its GUI — the headless workarounds below become optional.)

---

## Running it

All commands assume you are in the workspace root and have built the package. **First-time build:**

```bash
cd /workspaces/ros2-line-follower
colcon build
source install/setup.bash
```

### 1. Generate the training data and train the model

The dataset and trained model are **not** committed to the repo (they're generated artifacts). Recreate them from code — the fixed random seed makes the data identical every time:

```bash
# Generate 3000 augmented, auto-labelled images (left / center / right)
python3 src/line_follower/scripts/generate_dataset.py

# (Optional) visually inspect a sample montage
python3 src/line_follower/scripts/inspect_dataset.py

# Train the CNN (~a few minutes on CPU)
python3 src/line_follower/scripts/train.py

# (Optional) evaluate with a confusion matrix
python3 src/line_follower/scripts/evaluate.py
```

This produces `line_follower_model.pth` in the workspace root.

### 2. Launch the simulation and drive the robot

Open separate terminals for each:

```bash
# Terminal 1 — simulation (Gazebo + robot + camera). Wait ~40s for Gazebo to start.
ros2 launch line_follower simulation.launch.py

# Terminal 2 — the AI controller: camera → CNN → /cmd_vel
ros2 run line_follower inference_node
```

The robot will drive forward, follow the straight section, and steer through the bend.

---

## Visualizing the results

Because the Codespace is headless (no Gazebo GUI), there are three purpose-built ways to see what the robot is doing.

### A. Trajectory plot — *where the robot went*

Records `/odom` and plots the robot's path against the line track.

```bash
# With the sim and inference_node running, in a new terminal:
ros2 run line_follower trajectory_recorder
# Drive for 30–60s, then press Ctrl+C — saves trajectory.png
```

Open `trajectory.png` in the editor to see the path overlaid on the reference line.

### B. Annotated camera video — *what the CNN saw and decided*

Runs the same inference but draws the prediction, confidence, and a steering arrow onto each frame, saving an `.mp4`.

```bash
# With the sim running, in a new terminal:
ros2 run line_follower overlay_recorder
# Drive for 30–60s, then press Ctrl+C — saves overlay_video.mp4
```

Download the file (right-click in the VS Code explorer → Download) to play it.

### C. Live Foxglove dashboard — *everything, in real time*

Streams live topics (camera, robot model, plots) to the Foxglove web app in your browser.

```bash
# With the sim running, in a new terminal:
ros2 run foxglove_bridge foxglove_bridge
```

Then:
1. In the VS Code **PORTS** tab, forward port **8765** and set its visibility to **Public**.
2. Open <https://app.foxglove.dev> → **Open connection** → **Foxglove WebSocket**.
3. Connect to `wss://<your-codespace-name>-8765.app.github.dev`.

> ⚠️ A public port is reachable by anyone with the URL while it's open. Stop the bridge when you're done, and never expose sensitive services this way.

---

## How it works

```
Gazebo camera ──► /camera/image_raw ──► inference_node ──► /cmd_vel ──► diff-drive plugin ──► wheels
                                            │
                                     LineFollowerCNN
                                  (left / center / right)
```

- **Robot description** ([`urdf/robot.urdf.xacro`](src/line_follower/urdf/robot.urdf.xacro)) — a differential-drive chassis with two wheels, a caster, and a forward-down camera. Includes Gazebo plugins for the camera sensor and differential drive.
- **The CNN** ([`scripts/model.py`](src/line_follower/scripts/model.py)) — three convolution-and-pooling blocks that compress a 48×64 image into features, then two fully-connected layers that output three class scores. ~417k parameters.
- **Training data** ([`scripts/generate_dataset.py`](src/line_follower/scripts/generate_dataset.py)) — synthetic frames with a dark line at random positions **and angles**, plus brightness, noise, blur, and texture variation so the model learns robustness rather than memorising clean images. Labels are derived automatically from the line's position at the bottom of the frame.
- **The controller** ([`line_follower/inference_node.py`](src/line_follower/line_follower/inference_node.py)) — loads the trained model, classifies each frame, and publishes a `Twist`: drive straight on `center`, steer *toward* the line on `left`/`right`, and on low confidence pause briefly then rotate to search for the line.

> **Note on the class mapping:** the dataset folders are read alphabetically, so the label indices are `center=0, left=1, right=2`. The inference node hard-codes this order. If you change the classes, update both places.

---

## Project structure

```
ros2-line-follower/
├── .devcontainer/
│   └── devcontainer.json          # the reproducible environment definition
├── src/line_follower/
│   ├── line_follower/             # ROS2 nodes (installed, run with `ros2 run`)
│   │   ├── camera_node.py          # minimal camera subscriber (Phase 2)
│   │   ├── synthetic_camera.py     # synthetic frame publisher
│   │   ├── inference_node.py       # the AI controller
│   │   ├── overlay_recorder.py     # annotated-video visualization
│   │   ├── trajectory_recorder.py  # path-plot visualization
│   │   └── model.py                # CNN definition (copy of scripts/model.py)
│   ├── scripts/                   # standalone tools (run with `python3`)
│   │   ├── generate_dataset.py
│   │   ├── inspect_dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── urdf/robot.urdf.xacro
│   ├── worlds/line_track.world
│   ├── launch/simulation.launch.py
│   ├── package.xml
│   └── setup.py
└── README.md
```

Generated artifacts (`dataset/`, `*.pth`, `*.mp4`, `build/`, `install/`, `log/`) are git-ignored — regenerate them from the scripts above.

---

## Project phases

Each phase builds on the previous one. Follow them in order.

| Phase | Topic | What you build |
|-------|-------|----------------|
| 1 | Environment & dev container | A reproducible ROS2 + PyTorch Codespace, debugged from scratch |
| 2 | Robot & simulation | URDF robot, headless Gazebo, a working camera pipeline |
| 3 | AI vision & control | Dataset, CNN training, deployment, line following with recovery |
| 4 | CI/CD with GitHub Actions | Automated build & test on every push |
| 5 | Polish & prebuilds | Faster Codespace startup, one-click badge, release |

---

## Troubleshooting

Real issues encountered while building this, and their fixes — you may hit the same ones.

- **`ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'`** — ROS2 Humble is compiled against Python 3.10; don't upgrade Python. Keep the base image's 3.10 and don't add a Python-version dev-container feature.
- **`cv_bridge` / `_ARRAY_API not found`** — a NumPy 2.x vs 1.x conflict. `cv_bridge` is built against NumPy 1.x, so the project pins `numpy<2`.
- **Camera topic exists but `Publisher count: 0`** — the Gazebo camera plugin needs the robot spawned **from a file**, not from the `/robot_description` topic (which strips `<gazebo>` tags). The launch file writes the processed URDF to `/tmp` and spawns from there.
- **Gazebo camera produces no frames in a Codespace** — headless rendering needs a virtual display. The launch file starts `Xvfb` and forces software OpenGL (`LIBGL_ALWAYS_SOFTWARE=1`). Frame rate will be low (~8 Hz) without a GPU — this is expected.
- **`gzclient` exits with code -6** — the Gazebo GUI can't start without a display. This is harmless; the headless `gzserver` is what matters, and the launch file disables the GUI.
- **Robot drives backwards / turns the wrong way** — a wheel-joint axis sign issue in the URDF. The `<axis>` of the wheel joints is set to account for the cylinder rotation.
- **Foxglove won't connect** — make sure port 8765 is forwarded **and set to Public**, and use the `wss://` scheme (not `https://`) in the connection URL.

---

## License

Licensed under the Apache License 2.0 — see [`LICENSE`](LICENSE).

## Acknowledgements

Built with [ROS2 Humble](https://docs.ros.org/en/humble/), [PyTorch](https://pytorch.org/), [Gazebo](https://gazebosim.org/), and [Foxglove](https://foxglove.dev/), all running in [GitHub Codespaces](https://github.com/features/codespaces).
