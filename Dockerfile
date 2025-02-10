# Use Python 3.10 slim base image
FROM python:3.10-slim-bookworm AS build 

# Install required system packages
RUN apt update && apt install --no-install-recommends -y \
    curl \
    git \
    python3-opencv \
    build-essential \
    python3-pip \
    python3-venv \
    libglib2.0-0 \
    libgl1 \
    libturbojpeg0 \
    libgfortran5 \ 
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ARG POETRY_VERSION
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${POETRY_HOME}/bin:${PATH}"

# Ensure system TensorRT is used (disable pip TensorRT installation)
ENV NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=True

# Copy only files that are necessary to install dependencies
COPY poetry.lock poetry.toml pyproject.toml /code/

WORKDIR /code
RUN poetry install --no-root

# Copy the rest of the project
COPY . /code/

### **Final Image for Execution**
FROM python:3.10-slim-bookworm 
RUN apt update && apt install --no-install-recommends -y \
    libglib2.0-0 \
    libgl1 \
    libturbojpeg0 \
    libgfortran5 \  
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /code /code
# Set the LD_LIBRARY_PATH explicitly first
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/openblas-pthread
# # If needed, extend LD_LIBRARY_PATH later
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/aarch64-linux-gnu
ENV PYTHONPATH="/usr/lib/python3/dist-packages:/usr/lib/python3.10/dist-packages"


WORKDIR /code
ENV PATH="/code/.venv/bin:$PATH"
CMD [ "python", "main.py" ]
