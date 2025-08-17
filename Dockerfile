FROM nvcr.io/nvidia/pytorch:24.02-py3
# v2: python 3.10 포함

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    libstdc++6 \
    libgcc-s1 \
    tmux \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Set system Python as default
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace

# Upgrade pip and install dependencies with the right CUDA wheels
#RUN pip install -no-cache-dir —upgrade pip setuptools wheel
#RUN pip install --no-cache-dir --upgrade pip setuptools wheel
#RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
# && python -m pip install --no-cache-dir "protobuf<3.20" "numba>=0.57,<0.59"


CMD ["/bin/bash"]


## v2: docker build -t hyoyoon/pretrain-nlp:v2 .