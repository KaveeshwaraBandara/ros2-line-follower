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
# requirements-gpu.txt pins the CUDA 12.1 wheel index for torch (local GPU
# training only — CI and the devcontainer use plain requirements.txt instead)
COPY requirements.txt requirements-gpu.txt ./
RUN pip3 install --no-cache-dir -r requirements-gpu.txt \
    || (echo "Pinned CUDA torch version not found — falling back to latest" && \
        pip3 install --no-cache-dir \
            numpy==1.26.4 \
            opencv-python-headless==4.10.0.84 \
            matplotlib==3.10.9 \
            pytest==9.1.0 && \
        pip3 install --no-cache-dir torch \
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
