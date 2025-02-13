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
# Install Poetry for vitural env build up 

WORKDIR /code

# Copy dependency files first for caching
COPY requirements.txt /code/

RUN pip3 install --no-cache-dir -r requirements.txt

# Install torch/torchvision and onnxruntime sperately
RUN pip3 install --no-cache-dir\
    https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl

RUN pip3 uninstall -y numpy && pip3 install numpy==1.23.5

# Copy the rest of the project files
COPY . /code/

# # To allow virtial env able to use tensorrt
# COPY /usr/lib/python3.10/dist-packages/tensorrt /usr/lib/python3.10/dist-packages
# # To allow virtial env able to use latest version of libstdc++.so.6
# COPY /usr/lib/aarch64-linux-gnu/libstdc++.so.6 /usr/lib/aarch64-linux-gnu/libstdc++.so.6

# Run the detector
CMD ["python3", "main.py"]


# how to build sudo docker build -t mcvt_yq/object_detector_arm64:latest --platform linux/arm64 .