FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Base toolchain and runtime deps for rt_compile.sh / bqsim_rt.sh
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    ca-certificates \
    git \
    wget \
    curl \
    pkg-config \
    cmake \
    ninja-build \
    gcc-9 \
    g++-9 \
    libgomp1 \
    libopenmpi-dev \
    openmpi-bin \
    libeigen3-dev \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps used by scripts/tools
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir \
       numpy \
       qiskit==1.2.0 \
       qiskit-aer==0.15.0

# Quantum++ (qpp): required by BQSim/cuquantum_test/CMakeLists.txt find_package(qpp)
ARG QPP_REF=v5.1
RUN git clone --depth 1 --branch ${QPP_REF} https://github.com/softwareQinc/qpp.git /tmp/qpp \
    && cmake -S /tmp/qpp -B /tmp/qpp/build \
    && cmake --build /tmp/qpp/build -j \
    && cmake --install /tmp/qpp/build \
    && rm -rf /tmp/qpp

# cuQuantum (required by current CMake/app link settings)
ARG TARGETARCH
ARG CUQUANTUM_VERSION=24.11.0.21
ARG CUQUANTUM_CUDA_MAJOR=12
RUN set -eux; \
    ARCH="${TARGETARCH:-}"; \
    if [ -z "${ARCH}" ]; then ARCH="$(dpkg --print-architecture)"; fi; \
    case "${ARCH}" in \
      amd64|x86_64) CUQ_PLATFORM="linux-x86_64" ;; \
      arm64|aarch64) CUQ_PLATFORM="linux-sbsa" ;; \
      *) echo "Unsupported architecture for cuQuantum: ${ARCH}" >&2; exit 1 ;; \
    esac; \
    CUQ_DIR="cuquantum-${CUQ_PLATFORM}-${CUQUANTUM_VERSION}_cuda${CUQUANTUM_CUDA_MAJOR}-archive"; \
    CUQ_URL="https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/${CUQ_PLATFORM}/${CUQ_DIR}.tar.xz"; \
    wget -q "${CUQ_URL}" -O /tmp/cuquantum.tar.xz; \
    tar -xf /tmp/cuquantum.tar.xz -C /usr/local; \
    ln -sfn "/usr/local/${CUQ_DIR}" /usr/local/cuquantum; \
    rm -f /tmp/cuquantum.tar.xz

ENV CUQUANTUM_ROOT=/usr/local/cuquantum
ENV CUSTATEVEC_LIBRARY=${CUQUANTUM_ROOT}/lib/libcustatevec.so
ENV LD_LIBRARY_PATH=${CUQUANTUM_ROOT}/lib:${LD_LIBRARY_PATH}

# Defaults aligned with BQSim/rt_compile.sh (can be overridden at runtime)
# Leave CMAKE_CUDA_ARCHITECTURES unset: rt_compile.sh auto-detects GPU arch.
ENV CMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-9
ENV CMAKE_C_COMPILER=/usr/bin/gcc-9
ENV CMAKE_CXX_COMPILER=/usr/bin/g++-9

# Mount OptiX SDK include path here when running:
#   -v /path/to/NVIDIA-OptiX-SDK:/opt/optix:ro
ENV OptiX_INSTALL_DIR=/opt/optix

# Source code is expected via bind mount for live-edit workflow.
WORKDIR /workspace/RT_BQSim

CMD ["bash"]
