#!/usr/bin/env python3
"""GUI launcher for the ROS2 Line Follower simulation system."""

import enum
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, scrolledtext, ttk

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WS = "/workspaces/ros2-line-follower"
SOURCE_CMD = (
    "source /opt/ros/humble/setup.bash && "
    f"source {WS}/install/setup.bash"
)
SCRIPTS = f"{WS}/src/line_follower/scripts"
CONTAINER_FILTER = "ros2-line-follower"


# ---------------------------------------------------------------------------
# Process state
# ---------------------------------------------------------------------------

class ProcessStatus(enum.Enum):
    IDLE = "Idle"
    STARTING = "Starting..."
    RUNNING = "Running"
    ERROR = "Error"
    STOPPED = "Stopped"


STATUS_COLORS = {
    ProcessStatus.IDLE: "#9E9E9E",
    ProcessStatus.STARTING: "#FFC107",
    ProcessStatus.RUNNING: "#4CAF50",
    ProcessStatus.ERROR: "#F44336",
    ProcessStatus.STOPPED: "#9E9E9E",
}


# ---------------------------------------------------------------------------
# ManagedProcess
# ---------------------------------------------------------------------------

class ManagedProcess:
    def __init__(self):
        self.output_queue: queue.Queue = queue.Queue()
        self.status = ProcessStatus.IDLE
        self.returncode: int | None = None
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None

    def start(self, cmd: list[str], cwd: str | None = None):
        if self.is_running():
            return
        self.returncode = None
        self.status = ProcessStatus.STARTING
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd or PROJECT_ROOT,
            env=env,
            start_new_session=True,  # isolate child's process group from the GUI
        )
        self.status = ProcessStatus.RUNNING
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()

    def _read_output(self):
        proc = self._proc
        try:
            for line in proc.stdout:
                self.output_queue.put(line.rstrip())
        except Exception:
            pass
        proc.wait()
        self.returncode = proc.returncode
        self.status = ProcessStatus.ERROR if proc.returncode not in (0, -15, -2, None) else ProcessStatus.STOPPED
        self.output_queue.put(None)  # sentinel

    def stop(self):
        proc = self._proc
        if proc is None or proc.poll() is not None:
            self.status = ProcessStatus.STOPPED
            return
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
        self.status = ProcessStatus.STOPPED

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


# ---------------------------------------------------------------------------
# DockerManager
# ---------------------------------------------------------------------------

class DockerManager:
    def __init__(self):
        self._container_name: str | None = None

    def discover_container(self) -> str | None:
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={CONTAINER_FILTER}",
                 "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=5
            )
            names = [n.strip() for n in result.stdout.strip().splitlines() if n.strip()]
            if names:
                self._container_name = names[0]
                return self._container_name
        except Exception:
            pass
        self._container_name = None
        return None

    @property
    def container_name(self) -> str | None:
        return self._container_name

    def clear_container(self):
        self._container_name = None

    def run_cmd(self, inner: str) -> list[str]:
        # -T: no pseudo-TTY (we pipe stdout, TTY would conflict with signal delivery)
        # exec: replaces bash with the actual command so it becomes PID 1 inside the
        # container and receives docker kill signals directly instead of via bash.
        return [
            "docker", "compose", "run", "--rm", "-T", "ros2-line-follower",
            "bash", "-c", f"{SOURCE_CMD} && exec {inner}"
        ]

    def run_cmd_no_source(self, inner: str) -> list[str]:
        return [
            "docker", "compose", "run", "--rm", "-T", "ros2-line-follower",
            "bash", "-c", f"exec {inner}"
        ]

    def exec_cmd(self, inner: str) -> list[str]:
        c = self._container_name
        if not c:
            raise RuntimeError("No container name discovered yet")
        return [
            "docker", "exec", "-i", c,
            "bash", "-c",
            f"export PYTHONUNBUFFERED=1 && {SOURCE_CMD} && {inner}"
        ]

    def stop_all(self):
        # docker compose down only affects `docker compose up` services, not one-off
        # `docker compose run` containers. We must stop those explicitly.
        #
        # Signal strategy:
        #  1. SIGINT  → ros2 launch treats this as Ctrl+C and shuts down Gazebo cleanly
        #  2. 3-second grace period for Gazebo to close its window and exit
        #  3. SIGKILL → anything still running is force-killed immediately
        names = self._find_running_containers()
        for name in names:
            try:
                subprocess.run(
                    ["docker", "kill", "--signal=SIGINT", name],
                    capture_output=True, timeout=5
                )
            except Exception:
                pass
        if names:
            time.sleep(3)
        still_running = self._find_running_containers()
        for name in still_running:
            try:
                subprocess.run(
                    ["docker", "kill", name],
                    capture_output=True, timeout=5
                )
            except Exception:
                pass

    def _find_running_containers(self) -> list[str]:
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={CONTAINER_FILTER}",
                 "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=5
            )
            return [n.strip() for n in result.stdout.strip().splitlines() if n.strip()]
        except Exception:
            return []

    def preflight(self) -> dict:
        return {
            "docker_image": self._check_image(),
            "colcon_built": os.path.isdir(os.path.join(PROJECT_ROOT, "install", "line_follower")),
            "model_exists": os.path.isfile(os.path.join(PROJECT_ROOT, "line_follower_model.pth")),
            "dataset_exists": os.path.isdir(os.path.join(PROJECT_ROOT, "dataset", "center")),
        }

    def _check_image(self) -> bool:
        try:
            r = subprocess.run(
                ["docker", "images", "-q", "ros2-line-follower:latest"],
                capture_output=True, text=True, timeout=5
            )
            return bool(r.stdout.strip())
        except Exception:
            return False


# ---------------------------------------------------------------------------
# ProcessOutputPanel
# ---------------------------------------------------------------------------

class ProcessOutputPanel(ttk.LabelFrame):
    def __init__(self, parent, label: str, height: int = 8, **kw):
        super().__init__(parent, text=label, **kw)
        self._build(height)

    def _build(self, height: int):
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", pady=(2, 4))

        self._canvas = tk.Canvas(ctrl, width=14, height=14, highlightthickness=0)
        self._canvas.pack(side="left", padx=(4, 2))
        self._dot = self._canvas.create_oval(2, 2, 12, 12, fill="#9E9E9E", outline="")

        self._status_lbl = ttk.Label(ctrl, text="Idle", width=12)
        self._status_lbl.pack(side="left", padx=4)

        self.stop_btn = ttk.Button(ctrl, text="Stop", state="disabled", width=6)
        self.stop_btn.pack(side="right", padx=4)
        self.start_btn = ttk.Button(ctrl, text="Start", width=6)
        self.start_btn.pack(side="right", padx=2)

        self._text = scrolledtext.ScrolledText(
            self, state="disabled", height=height,
            font=("Courier", 9), bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white", wrap="word",
        )
        self._text.pack(fill="both", expand=True, padx=2, pady=(0, 2))

    def set_status(self, status: ProcessStatus):
        color = STATUS_COLORS[status]
        self._canvas.itemconfig(self._dot, fill=color)
        self._status_lbl.config(text=status.value)
        running = status == ProcessStatus.RUNNING
        starting = status == ProcessStatus.STARTING
        self.start_btn.config(state="disabled" if (running or starting) else "normal")
        self.stop_btn.config(state="normal" if (running or starting) else "disabled")

    def append_line(self, text: str):
        self._text.configure(state="normal")
        self._text.insert(tk.END, text + "\n")
        self._text.configure(state="disabled")
        self._text.see(tk.END)

    def clear(self):
        self._text.configure(state="normal")
        self._text.delete("1.0", tk.END)
        self._text.configure(state="disabled")

    def set_start_enabled(self, enabled: bool, tooltip: str = ""):
        self.start_btn.config(state="normal" if enabled else "disabled")
        if tooltip:
            self._status_lbl.config(text=tooltip if not enabled else "Idle")

    def bind_start(self, cb):
        self.start_btn.config(command=cb)

    def bind_stop(self, cb):
        self.stop_btn.config(command=cb)


# ---------------------------------------------------------------------------
# Setup Tab
# ---------------------------------------------------------------------------

class SetupTab(ttk.Frame):
    def __init__(self, parent, docker: DockerManager, procs: dict, output_panel: ProcessOutputPanel, **kw):
        super().__init__(parent, **kw)
        self._docker = docker
        self._procs = procs
        self._panel = output_panel
        self._active_key: str | None = None
        self._build()

    def _build(self):
        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(pane)
        pane.add(left, weight=1)

        right = ttk.Frame(pane)
        pane.add(right, weight=2)

        self._panel.master = right  # re-parent output panel
        # Output panel placed in right side
        out_frame = ttk.LabelFrame(right, text="Output")
        out_frame.pack(fill="both", expand=True, padx=4, pady=4)
        self._out_text = scrolledtext.ScrolledText(
            out_frame, state="disabled", height=20,
            font=("Courier", 9), bg="#1e1e1e", fg="#d4d4d4", wrap="word"
        )
        self._out_text.pack(fill="both", expand=True, padx=2, pady=2)

        ttk.Label(left, text="Setup Steps", font=("", 11, "bold")).pack(anchor="w", padx=4, pady=(4, 8))

        self._steps = [
            ("x11",          "1. Allow X11 Display (xhost +)",     None),
            ("docker_build", "2. Build Docker Image",               None),
            ("colcon_build", "3. Build ROS2 Package (colcon)",      "docker_build"),
            (None, None, None),  # separator
            ("dataset",      "4. Generate Dataset",                 "colcon_build"),
            ("train",        "5. Train Model",                      "dataset"),
            ("evaluate",     "6. Evaluate Model (optional)",        "train"),
            ("inspect",      "7. Inspect Dataset (optional)",       "dataset"),
        ]

        self._btns: dict[str, ttk.Button] = {}
        self._badges: dict[str, ttk.Label] = {}

        for key, label, dep in self._steps:
            if key is None:
                ttk.Separator(left, orient="horizontal").pack(fill="x", padx=4, pady=6)
                continue
            row = ttk.Frame(left)
            row.pack(fill="x", padx=4, pady=2)
            badge = ttk.Label(row, text="—", width=3, foreground="#9E9E9E")
            badge.pack(side="left")
            ttk.Label(row, text=label, anchor="w").pack(side="left", padx=4, fill="x", expand=True)
            btn = ttk.Button(row, text="Run", width=5, command=lambda k=key: self._run_step(k))
            btn.pack(side="right")
            self._btns[key] = btn
            self._badges[key] = badge

        self.update_badges()

    def update_badges(self):
        pf = self._docker.preflight()
        mapping = {
            "docker_build": pf["docker_image"],
            "colcon_build": pf["colcon_built"],
            "dataset":      pf["dataset_exists"],
            "train":        pf["model_exists"],
        }
        for key, done in mapping.items():
            if key in self._badges:
                self._badges[key].config(
                    text="✓" if done else "—",
                    foreground="#4CAF50" if done else "#9E9E9E"
                )
        # Disable train if dataset missing
        if "train" in self._btns:
            self._btns["train"].config(state="normal" if pf["dataset_exists"] else "disabled")

    def _run_step(self, key: str):
        if key in self._procs and self._procs[key].is_running():
            return
        self._active_key = key
        self._append_out(f"\n=== {key.upper()} ===\n")

        if key == "x11":
            proc = self._procs.get("x11") or ManagedProcess()
            self._procs["x11"] = proc
            proc.start(["xhost", "+"])
        elif key == "docker_build":
            self._procs["docker_build"].start(
                ["docker", "compose", "build"], cwd=PROJECT_ROOT
            )
        elif key == "colcon_build":
            self._procs["colcon_build"].start(
                self._docker.run_cmd("colcon build --symlink-install"),
                cwd=PROJECT_ROOT
            )
        elif key == "dataset":
            self._procs["dataset"].start(
                self._docker.run_cmd_no_source(f"python3 {SCRIPTS}/generate_dataset.py"),
                cwd=PROJECT_ROOT
            )
        elif key == "train":
            self._procs["train"].start(
                self._docker.run_cmd_no_source(f"python3 {SCRIPTS}/train.py"),
                cwd=PROJECT_ROOT
            )
        elif key == "evaluate":
            self._procs["evaluate"].start(
                self._docker.run_cmd_no_source(f"python3 {SCRIPTS}/evaluate.py"),
                cwd=PROJECT_ROOT
            )
        elif key == "inspect":
            self._procs["inspect"].start(
                self._docker.run_cmd_no_source(f"python3 {SCRIPTS}/inspect_dataset.py"),
                cwd=PROJECT_ROOT
            )

    def drain_output(self):
        if self._active_key and self._active_key in self._procs:
            proc = self._procs[self._active_key]
            while not proc.output_queue.empty():
                line = proc.output_queue.get_nowait()
                if line is not None:
                    self._append_out(line)

    def _append_out(self, text: str):
        self._out_text.configure(state="normal")
        self._out_text.insert(tk.END, text + "\n")
        self._out_text.configure(state="disabled")
        self._out_text.see(tk.END)


# ---------------------------------------------------------------------------
# Simulation Tab
# ---------------------------------------------------------------------------

class SimulationTab(ttk.Frame):
    def __init__(self, parent, docker: DockerManager, proc: ManagedProcess,
                 discovery_queue: queue.Queue, nodes_tab_ref, **kw):
        super().__init__(parent, **kw)
        self._docker = docker
        self._proc = proc
        self._discovery_q = discovery_queue
        self._nodes_ref = nodes_tab_ref  # set after NodesTab is created
        self._build()

    def _build(self):
        # Config section
        cfg = ttk.LabelFrame(self, text="Configuration")
        cfg.pack(fill="x", padx=8, pady=(8, 4))

        ttk.Label(cfg, text="World:").grid(row=0, column=0, padx=8, pady=6, sticky="w")
        self._world_var = tk.StringVar(value="line_track")
        self._world_cb = ttk.Combobox(cfg, textvariable=self._world_var, width=18, state="readonly",
                                       values=["line_track", "rectangle", "triangle",
                                               "corridor_maze", "curved", "zigzag", "messy"])
        self._world_cb.grid(row=0, column=1, padx=4, pady=6, sticky="w")

        self._gui_var = tk.BooleanVar(value=False)
        self._gui_chk = ttk.Checkbutton(cfg, text="Enable Gazebo GUI", variable=self._gui_var)
        self._gui_chk.grid(row=0, column=2, padx=16, pady=6, sticky="w")

        self._gui_hint = ttk.Label(cfg, text="(xhost + will be run automatically)",
                                    foreground="#888888", font=("", 8))
        self._gui_hint.grid(row=0, column=3, padx=4)
        self._gui_var.trace_add("write", lambda *_: self._gui_hint.config(
            foreground="#888888" if not self._gui_var.get() else "#FFC107"
        ))

        btn_frame = ttk.Frame(cfg)
        btn_frame.grid(row=1, column=0, columnspan=4, pady=(0, 8), padx=8, sticky="w")
        self._start_btn = ttk.Button(btn_frame, text="Start Simulation", command=self._start_sim, width=16)
        self._start_btn.pack(side="left", padx=(0, 8))
        self._stop_btn = ttk.Button(btn_frame, text="Stop Simulation", command=self._stop_sim,
                                     width=16, state="disabled")
        self._stop_btn.pack(side="left")

        self._colcon_warn = ttk.Label(self, text="⚠ Colcon build required — go to Setup tab first",
                                       foreground="#FFC107", font=("", 9, "bold"))

        # Output panel
        self._panel = ProcessOutputPanel(self, "Simulation Output", height=22)
        self._panel.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self._panel.start_btn.pack_forget()
        self._panel.stop_btn.pack_forget()

    def check_colcon(self, built: bool):
        if built:
            self._colcon_warn.pack_forget()
            self._start_btn.config(state="normal")
        else:
            self._colcon_warn.pack(fill="x", padx=8, pady=2)
            self._start_btn.config(state="disabled")

    def _start_sim(self):
        if self._gui_var.get():
            subprocess.Popen(["xhost", "+"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self._panel.clear()
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="disabled")
        self._world_cb.config(state="disabled")
        self._gui_chk.config(state="disabled")

        # Kill any orphaned containers before launching so we don't accumulate them
        threading.Thread(target=self._cleanup_then_start, daemon=True).start()

    def _cleanup_then_start(self):
        orphans = self._docker._find_running_containers()
        if orphans:
            self.after(0, lambda: self._panel.append_line(
                f"--- Stopping {len(orphans)} orphaned container(s) first ---"
            ))
            for name in orphans:
                try:
                    subprocess.run(["docker", "stop", name],
                                   capture_output=True, timeout=20)
                except Exception:
                    pass
        self.after(0, self._launch_proc)

    def _launch_proc(self):
        world = self._world_var.get()
        gui = "true" if self._gui_var.get() else "false"
        inner = f"ros2 launch line_follower simulation.launch.py world_name:={world} gui:={gui}"
        cmd = self._docker.run_cmd(inner)

        self._proc.start(cmd, cwd=PROJECT_ROOT)
        self._panel.set_status(ProcessStatus.RUNNING)
        self._stop_btn.config(state="normal")

        t = threading.Thread(target=self._discover_loop, daemon=True)
        t.start()

    def _discover_loop(self):
        for _ in range(30):
            name = self._docker.discover_container()
            if name:
                self._discovery_q.put(name)
                return
            time.sleep(2)
        self._discovery_q.put(None)  # timed out

    def _stop_sim(self):
        nodes = self._nodes_ref
        if nodes and (nodes.inference_running() or nodes.recording_running()):
            if not messagebox.askokcancel(
                "Stop Simulation",
                "Inference or recording is still running.\n"
                "Stopping the simulation now may lose unsaved output files.\n\n"
                "Stop anyway?"
            ):
                return

        # Disable buttons immediately so the user can't double-click
        self._stop_btn.config(state="disabled")
        self._start_btn.config(state="disabled")
        self._panel.append_line("\n--- Stopping simulation ---")
        if nodes:
            nodes.on_container_lost()

        # Run the blocking Docker cleanup in a background thread
        threading.Thread(target=self._do_stop, daemon=True).start()

    def _do_stop(self):
        self._proc.stop()
        self._docker.stop_all()
        self._docker.clear_container()
        # Schedule UI restore on the main thread
        self.after(0, self._on_stopped)

    def _on_stopped(self):
        self._panel.set_status(ProcessStatus.STOPPED)
        self._panel.append_line("--- Simulation stopped ---")
        self._world_cb.config(state="readonly")
        self._gui_chk.config(state="normal")
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")

    def drain_output(self):
        while not self._proc.output_queue.empty():
            line = self._proc.output_queue.get_nowait()
            if line is None:
                rc = self._proc.returncode
                if rc not in (0, None, -15, -2):
                    self._panel.set_status(ProcessStatus.ERROR)
                else:
                    self._panel.set_status(ProcessStatus.STOPPED)
                self._world_cb.config(state="readonly")
                self._gui_chk.config(state="normal")
                self._start_btn.config(state="normal")
                self._stop_btn.config(state="disabled")
            else:
                self._panel.append_line(line)


# ---------------------------------------------------------------------------
# Nodes Tab
# ---------------------------------------------------------------------------

class NodesTab(ttk.Frame):
    def __init__(self, parent, docker: DockerManager,
                 procs: dict, **kw):
        super().__init__(parent, **kw)
        self._docker = docker
        self._procs = procs
        self._container_ready = False
        self._pending_save: tuple[str, str] | None = None  # (dest_path, default_filename)
        self._build()

    def _build(self):
        pane = ttk.PanedWindow(self, orient="vertical")
        pane.pack(fill="both", expand=True, padx=8, pady=8)

        # Inference
        inf_frame = ttk.LabelFrame(self, text="Inference Node")
        pane.add(inf_frame, weight=1)
        self._no_model_lbl = ttk.Label(
            inf_frame, text="Model not found — run Train in Setup tab",
            foreground="#F44336", font=("", 9)
        )
        self._inf_panel = ProcessOutputPanel(inf_frame, "inference_node", height=7)
        self._inf_panel.pack(fill="both", expand=True, padx=4, pady=4)
        self._inf_panel.bind_start(self._start_inference)
        self._inf_panel.bind_stop(self._stop_inference)

        # Recording
        rec_frame = ttk.LabelFrame(self, text="Recording")
        pane.add(rec_frame, weight=1)
        radio_row = ttk.Frame(rec_frame)
        radio_row.pack(fill="x", padx=4, pady=(4, 0))
        self._rec_mode = tk.StringVar(value="trajectory")
        self._rad_traj = ttk.Radiobutton(radio_row, text="Trajectory (trajectory.png)",
                                          variable=self._rec_mode, value="trajectory")
        self._rad_traj.pack(side="left", padx=8)
        self._rad_video = ttk.Radiobutton(radio_row, text="Video Overlay (overlay_video.mp4)",
                                           variable=self._rec_mode, value="overlay")
        self._rad_video.pack(side="left", padx=8)
        self._rec_panel = ProcessOutputPanel(rec_frame, "recorder", height=7)
        self._rec_panel.pack(fill="both", expand=True, padx=4, pady=4)
        self._rec_panel.bind_start(self._start_recording)
        self._rec_panel.bind_stop(self._stop_recording)
        self._rec_saved_lbl = ttk.Label(rec_frame, text="", foreground="#4CAF50", font=("", 9))
        self._rec_saved_lbl.pack(pady=(0, 4))

        # Foxglove
        fox_frame = ttk.LabelFrame(self, text="Foxglove Bridge")
        pane.add(fox_frame, weight=1)
        fox_info = ttk.Frame(fox_frame)
        fox_info.pack(fill="x", padx=4, pady=(4, 0))
        self._fox_link = ttk.Label(fox_info, text="", foreground="#2196F3", cursor="hand2", font=("", 9))
        self._fox_link.pack(side="left", padx=4)
        self._fox_link.bind("<Button-1>", lambda _: webbrowser.open("https://app.foxglove.dev"))
        self._fox_panel = ProcessOutputPanel(fox_frame, "foxglove_bridge", height=7)
        self._fox_panel.pack(fill="both", expand=True, padx=4, pady=4)
        self._fox_panel.bind_start(self._start_foxglove)
        self._fox_panel.bind_stop(self._stop_foxglove)

        self.on_container_lost()  # initial state: all disabled

    def on_container_ready(self, container_name: str):
        self._container_ready = True
        self._update_inference_btn()
        self._rec_panel.set_start_enabled(True)
        self._fox_panel.set_start_enabled(True)

    def on_container_lost(self):
        self._container_ready = False
        self._inf_panel.set_start_enabled(False, "Waiting for container...")
        self._rec_panel.set_start_enabled(False, "Waiting for container...")
        self._fox_panel.set_start_enabled(False, "Waiting for container...")
        self._fox_link.config(text="")

    def update_model_status(self, model_exists: bool):
        if model_exists:
            self._no_model_lbl.pack_forget()
        else:
            self._no_model_lbl.pack(before=self._inf_panel, fill="x", padx=4, pady=2)
        self._update_inference_btn()

    def _update_inference_btn(self):
        model_ok = os.path.isfile(os.path.join(PROJECT_ROOT, "line_follower_model.pth"))
        enabled = self._container_ready and model_ok and not self._procs["inference"].is_running()
        self._inf_panel.set_start_enabled(enabled)

    def inference_running(self) -> bool:
        return self._procs["inference"].is_running()

    def recording_running(self) -> bool:
        return self._procs["recording"].is_running()

    def _start_inference(self):
        cmd = self._docker.exec_cmd("ros2 run line_follower inference_node")
        self._inf_panel.clear()
        self._procs["inference"].start(cmd, cwd=PROJECT_ROOT)

    def _stop_inference(self):
        self._procs["inference"].stop()

    def _start_recording(self):
        mode = self._rec_mode.get()
        exe = "trajectory_recorder" if mode == "trajectory" else "overlay_recorder"
        cmd = self._docker.exec_cmd(f"ros2 run line_follower {exe}")
        self._rec_panel.clear()
        self._rec_saved_lbl.config(text="")
        self._procs["recording"].start(cmd, cwd=PROJECT_ROOT)
        self._rad_traj.config(state="disabled")
        self._rad_video.config(state="disabled")

    def _stop_recording(self):
        mode = self._rec_mode.get()
        is_video = mode == "overlay"
        default_name = "overlay_video.mp4" if is_video else "trajectory.png"
        ext = ".mp4" if is_video else ".png"
        filetypes = (
            [("MP4 video", "*.mp4"), ("All files", "*.*")]
            if is_video else
            [("PNG image", "*.png"), ("All files", "*.*")]
        )
        save_path = filedialog.asksaveasfilename(
            title="Save recording as...",
            initialdir=PROJECT_ROOT,
            initialfile=default_name,
            defaultextension=ext,
            filetypes=filetypes,
        )
        if not save_path:
            return  # user cancelled — recording keeps running
        self._pending_save = (save_path, default_name)
        self._rec_panel.append_line(f"\n--- Stopping recorder, saving to: {save_path} ---")
        self._procs["recording"].stop()

    def _start_foxglove(self):
        cmd = self._docker.exec_cmd("ros2 run foxglove_bridge foxglove_bridge")
        self._fox_panel.clear()
        self._procs["foxglove"].start(cmd, cwd=PROJECT_ROOT)

    def _stop_foxglove(self):
        self._procs["foxglove"].stop()

    def drain_output(self):
        # Inference
        self._drain_proc("inference", self._inf_panel, on_stop=lambda: (
            self._update_inference_btn(),
        ))
        # Recording
        def on_rec_stop():
            self._rad_traj.config(state="normal")
            self._rad_video.config(state="normal")
            pending = self._pending_save
            self._pending_save = None
            if pending:
                dest, default_name = pending
                src = os.path.join(PROJECT_ROOT, default_name)
                if os.path.exists(src):
                    try:
                        if os.path.abspath(dest) != os.path.abspath(src):
                            shutil.copy2(src, dest)
                        self._rec_saved_lbl.config(
                            text=f"Saved: {dest}", foreground="#4CAF50"
                        )
                    except Exception as e:
                        self._rec_saved_lbl.config(
                            text=f"Copy failed: {e}", foreground="#F44336"
                        )
                else:
                    self._rec_saved_lbl.config(
                        text=f"File not found: {src}", foreground="#F44336"
                    )
            else:
                mode = self._rec_mode.get()
                fname = "trajectory.png" if mode == "trajectory" else "overlay_video.mp4"
                self._rec_saved_lbl.config(
                    text=f"Saved: {os.path.join(PROJECT_ROOT, fname)}",
                    foreground="#4CAF50"
                )
        self._drain_proc("recording", self._rec_panel, on_stop=on_rec_stop)
        # Foxglove
        def on_fox_start():
            self._fox_link.config(text="Connect at ws://localhost:8765 → app.foxglove.dev")
        def on_fox_stop():
            self._fox_link.config(text="")
        self._drain_proc_with_start("foxglove", self._fox_panel,
                                     on_start=on_fox_start, on_stop=on_fox_stop)

    def _drain_proc(self, key: str, panel: ProcessOutputPanel, on_stop=None):
        proc = self._procs[key]
        while not proc.output_queue.empty():
            line = proc.output_queue.get_nowait()
            if line is None:
                panel.set_status(proc.status)
                if on_stop:
                    on_stop()
            else:
                panel.append_line(line)
                panel.set_status(proc.status)

    def _drain_proc_with_start(self, key: str, panel: ProcessOutputPanel,
                                on_start=None, on_stop=None):
        proc = self._procs[key]
        was_idle = proc.status in (ProcessStatus.IDLE, ProcessStatus.STOPPED)
        while not proc.output_queue.empty():
            line = proc.output_queue.get_nowait()
            if line is None:
                panel.set_status(proc.status)
                if on_stop:
                    on_stop()
            else:
                if was_idle and on_start:
                    on_start()
                    was_idle = False
                panel.append_line(line)
                panel.set_status(proc.status)


# ---------------------------------------------------------------------------
# Status Bar
# ---------------------------------------------------------------------------

class StatusBar(ttk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, relief="sunken", **kw)
        self._container_lbl = ttk.Label(self, text="Container: none", width=45, anchor="w")
        self._container_lbl.pack(side="left", padx=8, pady=2)
        ttk.Separator(self, orient="vertical").pack(side="left", fill="y", padx=4)
        self._colcon_lbl = ttk.Label(self, text="Colcon: —", foreground="#9E9E9E")
        self._colcon_lbl.pack(side="left", padx=6)
        ttk.Separator(self, orient="vertical").pack(side="left", fill="y", padx=4)
        self._model_lbl = ttk.Label(self, text="Model: —", foreground="#9E9E9E")
        self._model_lbl.pack(side="left", padx=6)
        ttk.Separator(self, orient="vertical").pack(side="left", fill="y", padx=4)
        self._dataset_lbl = ttk.Label(self, text="Dataset: —", foreground="#9E9E9E")
        self._dataset_lbl.pack(side="left", padx=6)

    def update(self, pf: dict, container: str | None):
        self._container_lbl.config(
            text=f"Container: {container[:40] if container else 'none'}",
            foreground="#4CAF50" if container else "#9E9E9E"
        )

        def badge(ok: bool) -> tuple[str, str]:
            return ("✓", "#4CAF50") if ok else ("✗", "#F44336")

        t, c = badge(pf["colcon_built"])
        self._colcon_lbl.config(text=f"Colcon: {t}", foreground=c)
        t, c = badge(pf["model_exists"])
        self._model_lbl.config(text=f"Model: {t}", foreground=c)
        t, c = badge(pf["dataset_exists"])
        self._dataset_lbl.config(text=f"Dataset: {t}", foreground=c)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ROS2 Line Follower Launcher")
        self.root.geometry("1150x800")
        self.root.minsize(900, 600)

        self._docker = DockerManager()
        self._discovery_q: queue.Queue = queue.Queue()

        self._procs = {
            "x11":          ManagedProcess(),
            "docker_build": ManagedProcess(),
            "colcon_build": ManagedProcess(),
            "dataset":      ManagedProcess(),
            "train":        ManagedProcess(),
            "evaluate":     ManagedProcess(),
            "inspect":      ManagedProcess(),
            "simulation":   ManagedProcess(),
            "inference":    ManagedProcess(),
            "recording":    ManagedProcess(),
            "foxglove":     ManagedProcess(),
        }

        self._build_ui()
        self._check_existing_container()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll)

    def _build_ui(self):
        self._status_bar = StatusBar(self.root)
        self._status_bar.pack(fill="x", side="bottom")

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", side="bottom")

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        setup_frame = ttk.Frame(nb)
        sim_frame = ttk.Frame(nb)
        nodes_frame = ttk.Frame(nb)

        nb.add(setup_frame, text="  Setup  ")
        nb.add(sim_frame, text="  Simulation  ")
        nb.add(nodes_frame, text="  Nodes  ")

        # Nodes tab must be created before SimulationTab so we can pass a ref
        self._nodes_tab = NodesTab(nodes_frame, self._docker, self._procs)
        self._nodes_tab.pack(fill="both", expand=True)

        self._sim_tab = SimulationTab(
            sim_frame, self._docker, self._procs["simulation"],
            self._discovery_q, self._nodes_tab
        )
        self._sim_tab.pack(fill="both", expand=True)

        # Dummy output panel passed to SetupTab (not used directly, output is internal)
        dummy = ProcessOutputPanel(setup_frame, "")
        self._setup_tab = SetupTab(setup_frame, self._docker, self._procs, dummy)
        self._setup_tab.pack(fill="both", expand=True)

    def _check_existing_container(self):
        def _check():
            name = self._docker.discover_container()
            if name:
                self._discovery_q.put(name)
        threading.Thread(target=_check, daemon=True).start()

    def _poll(self):
        # Drain discovery queue
        while not self._discovery_q.empty():
            name = self._discovery_q.get_nowait()
            if name:
                self._nodes_tab.on_container_ready(name)
            else:
                # Discovery timed out but sim process still running — keep polling
                pass

        # Drain setup tab outputs
        self._setup_tab.drain_output()

        # Drain simulation
        self._sim_tab.drain_output()

        # Drain nodes
        self._nodes_tab.drain_output()

        # Update status bar
        pf = self._docker.preflight()
        self._status_bar.update(pf, self._docker.container_name)

        # Update sim tab colcon warning
        self._sim_tab.check_colcon(pf["colcon_built"])

        # Update setup badges
        self._setup_tab.update_badges()

        # Update model status in nodes tab
        self._nodes_tab.update_model_status(pf["model_exists"])

        self.root.after(100, self._poll)

    def _on_close(self):
        any_running = any(p.is_running() for p in self._procs.values())
        if any_running:
            if not messagebox.askokcancel(
                "Quit",
                "Some processes are still running.\n"
                "Stopping them now may lose unsaved data.\n\nQuit anyway?"
            ):
                return
        for p in self._procs.values():
            try:
                p.stop()
            except Exception:
                pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()
