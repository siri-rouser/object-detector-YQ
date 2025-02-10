# Use NVIDIA's JetPack 6.2 base image for Jetson devices
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

# Set environment variables for optimal container behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DEBIAN_FRONTEND=noninteractive 

ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/


ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb .
RUN dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    git python3-pip libopenmpi-dev libopenblas-base libomp-dev libcusparselt0 libcusparselt-dev python3.10-venv\
    && rm -rf /var/lib/apt/lists/*

# Install Poetry (Python dependency management)
ARG POETRY_VERSION
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${POETRY_HOME}/bin:${PATH}"

# Copy only necessary files for dependency installation
COPY poetry.lock poetry.toml pyproject.toml /code/
WORKDIR /code
# Install Python dependencies using Poetry
RUN poetry install --no-root

# Set up Python virtual environment

RUN python3 -m venv /code/.venv
ENV PATH="/code/.venv/bin:$PATH"
# Ensure system TensorRT is used (disable pip TensorRT installation)
ENV NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=True

# Pip install onnxruntime-gpu, torch, torchvision and ultralytics
RUN python3 -m pip install --upgrade pip uv
RUN pip install --system \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl

# Remove build files to reduce container size
RUN rm -rf /root/.cache /root/.config/Ultralytics/persistent_cache.json

# Copy the rest of the project files
COPY . /code/

# # Set library paths for CUDA, OpenCV, and OpenBLAS
# ENV LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH"
# ENV PYTHONPATH="/usr/lib/python3/dist-packages:/usr/lib/python3.10/dist-packages"

# Set the working directory
WORKDIR /code

# Run the detector
CMD ["python", "main.py"]
