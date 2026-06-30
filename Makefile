PROJECT_ROOT := $(shell pwd)
WS           := /workspaces/ros2-line-follower
SCRIPTS      := $(WS)/src/line_follower/scripts
SOURCE       := source /opt/ros/humble/setup.bash && source $(WS)/install/setup.bash

# Find the running container name and exec a command inside it.
# Usage: $(call EXEC, <command>)
define EXEC
	@CNAME=$$(docker ps --filter "name=ros2-line-follower" --format "{{.Names}}" | head -1); \
	if [ -z "$$CNAME" ]; then \
		echo "ERROR: No running container found."; \
		echo "       Run 'make sim' (or another sim variant) in a separate terminal first."; \
		exit 1; \
	fi; \
	docker exec -i "$$CNAME" bash -c "export PYTHONUNBUFFERED=1 && $(SOURCE) && $(1)"
endef

.DEFAULT_GOAL := help
.PHONY: help gui build colcon-build dataset train evaluate inspect \
        sim sim-gui sim-rect sim-maze \
        inference record-traj record-video foxglove stop

## ── Help ────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  ROS2 Line Follower — available make targets"
	@echo ""
	@echo "  LAUNCHER"
	@echo "    make gui            Launch the graphical launcher (recommended)"
	@echo ""
	@echo "  ONE-TIME SETUP"
	@echo "    make build          Build the Docker image"
	@echo "    make colcon-build   Build the ROS2 package inside the container"
	@echo ""
	@echo "  DATASET & TRAINING  (run independently, no simulation needed)"
	@echo "    make dataset        Generate training dataset (~3000 images)"
	@echo "    make train          Train the CNN model"
	@echo "    make evaluate       Evaluate model accuracy (optional)"
	@echo "    make inspect        Save dataset preview image (optional)"
	@echo ""
	@echo "  SIMULATION  (run in Terminal A — blocks until Ctrl+C)"
	@echo "    make sim            Headless, default world (line_track)"
	@echo "    make sim-gui        With Gazebo window, default world"
	@echo "    make sim-rect       Headless, rectangle world"
	@echo "    make sim-maze       Headless, corridor_maze world"
	@echo ""
	@echo "  NODES  (run in separate terminals while simulation is running)"
	@echo "    make inference      Run the AI inference node"
	@echo "    make record-traj    Record trajectory → trajectory.png"
	@echo "    make record-video   Record annotated video → overlay_video.mp4"
	@echo "    make foxglove       Foxglove bridge (connect at ws://localhost:8765)"
	@echo ""
	@echo "  STOP"
	@echo "    make stop           Stop all containers (docker compose down)"
	@echo ""

## ── Launcher ────────────────────────────────────────────────────────────────

gui:
	python3 $(PROJECT_ROOT)/gui_launcher.py

## ── One-time setup ──────────────────────────────────────────────────────────

build:
	docker compose build

colcon-build:
	docker compose run --rm ros2-line-follower bash -c \
		"$(SOURCE) && colcon build --symlink-install"

## ── Dataset & Training ──────────────────────────────────────────────────────
# These use docker compose run (fresh container) so they don't require the
# simulation to be running first.

dataset:
	docker compose run --rm ros2-line-follower bash -c \
		"python3 $(SCRIPTS)/generate_dataset.py"

train:
	docker compose run --rm ros2-line-follower bash -c \
		"python3 $(SCRIPTS)/train.py"

evaluate:
	docker compose run --rm ros2-line-follower bash -c \
		"python3 $(SCRIPTS)/evaluate.py"

inspect:
	docker compose run --rm ros2-line-follower bash -c \
		"python3 $(SCRIPTS)/inspect_dataset.py"

## ── Simulation ──────────────────────────────────────────────────────────────

sim:
	@echo "Starting headless simulation (world: line_track). Press Ctrl+C to stop."
	docker compose run --rm ros2-line-follower bash -c \
		"$(SOURCE) && ros2 launch line_follower simulation.launch.py"

sim-gui:
	@echo "Allowing X11 access for Gazebo GUI..."
	xhost +
	docker compose run --rm ros2-line-follower bash -c \
		"$(SOURCE) && ros2 launch line_follower simulation.launch.py gui:=true"

sim-rect:
	@echo "Starting headless simulation (world: rectangle). Press Ctrl+C to stop."
	docker compose run --rm ros2-line-follower bash -c \
		"$(SOURCE) && ros2 launch line_follower simulation.launch.py world_name:=rectangle"

sim-maze:
	@echo "Starting headless simulation (world: corridor_maze). Press Ctrl+C to stop."
	docker compose run --rm ros2-line-follower bash -c \
		"$(SOURCE) && ros2 launch line_follower simulation.launch.py world_name:=corridor_maze"

## ── Nodes (exec into running container) ─────────────────────────────────────

inference:
	$(call EXEC,ros2 run line_follower inference_node)

record-traj:
	@echo "Recording trajectory. Press Ctrl+C to stop and save trajectory.png."
	$(call EXEC,ros2 run line_follower trajectory_recorder)

record-video:
	@echo "Recording video overlay. Press Ctrl+C to stop and save overlay_video.mp4."
	$(call EXEC,ros2 run line_follower overlay_recorder)

foxglove:
	@echo "Starting Foxglove bridge. Connect at: ws://localhost:8765"
	@echo "Open https://app.foxglove.dev → Open Connection → WebSocket"
	$(call EXEC,ros2 run foxglove_bridge foxglove_bridge)

## ── Stop ────────────────────────────────────────────────────────────────────

stop:
	@echo "Stopping all ros2-line-follower containers..."
	docker compose down
	@echo "Done."
