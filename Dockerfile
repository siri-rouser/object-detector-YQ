# Use NVIDIA's JetPack 6.2 base image for Jetson devices
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

# Set environment variables for optimal container behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt update && apt install --no-install-recommends -y \
    curl \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libblas3 \
    libjpeg-turbo8 \
    libomp-dev \
    python3-venv \
    python3-pip \
    tzdata

# Install dependencies especially for cuda_keyring
ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb .
RUN dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libopenmpi-dev libopenblas-base libomp-dev libcusparselt0 libcusparselt-dev libturbojpeg\
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y git

WORKDIR /code

# Copy dependency files first for caching
COPY requirements.txt /code/

RUN pip3 install --no-cache-dir -r requirements.txt

# Install torch/torchvision and onnxruntime sperately
RUN pip3 install --no-cache-dir\
    https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl

RUN apt install cmake -y
# Copy the rest of the project files
COPY . /code/

RUN git clone --branch 4.10.0 https://github.com/opencv/opencv.git /opt/opencv
RUN git clone --branch 4.10.0 https://github.com/opencv/opencv_contrib.git /opt/opencv_contrib
WORKDIR /opt/opencv

# Uninstall the pre-installed opencv-python package by ultralytics
RUN pip3 uninstall -y opencv-python

RUN mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
          -D WITH_CUDA=ON \
          -D WITH_CUDNN=ON \
          -D OPENCV_DNN_CUDA=ON \
          -D CUDA_ARCH_BIN=87 \
          -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
          .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

RUN pip3 uninstall -y numpy && pip3 install numpy==1.23.5

# Do this after all previous steps to avoid cache invalidation
RUN apt-get update && apt-get remove --purge -y 'libnvinfer*' 'python3-libnvinfer*' && \
    rm -rf /var/lib/apt/lists/*

# 1) Ensure /tmp is writable
RUN mkdir -p /tmp && chmod 1777 /tmp

# 2) Install apt-transport-https, gnupg, etc.
RUN apt-get update && apt-get install -y \
    apt-transport-https gnupg2 ca-certificates

RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer-dev=10.7.0.23-1+cuda12.6 \
    libnvinfer-plugin-dev=10.7.0.23-1+cuda12.6 \
    libnvinfer-bin=10.7.0.23-1+cuda12.6 \
    python3-libnvinfer=10.7.0.23-1+cuda12.6 \
    python3-libnvinfer-dev=10.7.0.23-1+cuda12.6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code
# Run the detector
CMD ["python3", "main.py"]

# how to build sudo docker build -t mcvt_yq/object_detector_arm64:latest --platform linux/arm64 .
# sudo docker run --runtime=nvidia -it \
#   -v /home/yuqiang/yl4300/Multi-Camera-Vision-Pipeline-YQ/docker-compose/object-detector/object-detector.settings.yaml:/code/settings.yaml \
#   -v /home/yuqiang/yl4300/Multi-Camera-Vision-Pipeline-YQ/subpackages/object-detector-YQ/object-detector-YQ/yolo11l.engine:/code/yolo11l.engine \
#   mcvt_yq/object_detector_arm64:v2.0 sh
