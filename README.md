# Debian-PyTorch-RTX5070Ti - Deploying vLLM with CUDA 12.8, FP4 / AWQ Quantization, FlashInfer, and Reasoning Mode
A concise, end‑to‑end tutorial for Debian 12 that builds a CUDA 12.8‑optimized PyTorch image and launches vLLM with FlashInfer and AWQ/FP4‑quantized models on NVIDIA Blackwell GPUs via Docker or Podman. The steps apply to the **RTX 5070 Ti** or any newer Blackwell-based GPU.

In this doc, I’ll show:

- Installing CUDA 12.8 & custom PyTorch
- Building a Docker image (or Podman) with **FlashInfer** as the flash attention backend
- Running AWQ/FP4-quantized models (like `QwQ-32B-AWQ`, `WhiteRabbitNeo-13B-AWQ`, `DeepSeek-R1-FP4`, or `TinyLlama`)
- Enabling **reasoning** outputs and **function/tool usage** with the new Qwen-based QwQ model
- Fine-tuning memory usage with `--gpu-memory-utilization`, `--cpu-offload-gb`, and `--max-model-len`

---

## 1. System Prerequisites

1. **NVIDIA Drivers** ≥ 545
2. **CUDA 12.8** installed
3. **Debian 12**, or a similar distro with a recent kernel
4. **Docker or Podman** plus **nvidia-container-toolkit** for GPU pass-through
5. [**Hugging Face Hub Token**](https://huggingface.co/settings/tokens) if you’re pulling private or gated models

### Quick Start for Debian 12

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential dkms linux-headers-$(uname -r) curl ca-certificates
```

#### Blacklist Nouveau and Install NVIDIA Drivers

```bash
echo 'blacklist nouveau' | sudo tee /etc/modprobe.d/blacklist-nvidia-nouveau.conf
sudo update-initramfs -u
```

Set up the CUDA 12.x repository and install:

```bash
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/3bf863cc.pub \
  | sudo gpg --dearmor -o /etc/apt/keyrings/nvidia-drivers.gpg

echo 'deb [signed-by=/etc/apt/keyrings/nvidia-drivers.gpg] \
  https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /' \
  | sudo tee /etc/apt/sources.list.d/nvidia-cuda.list

sudo apt update
sudo apt install -y nvidia-driver cuda
sudo reboot now
```

Check with `nvidia-smi`.

---

## 2. Build PyTorch (CUDA 12.8 + AVX-512)

> **Note**: You can skip this if you’re okay with [PyTorch Nightly wheels](https://download.pytorch.org/whl/nightly/cu128). However, if you need full custom build flags or AVX-512, do this:

```bash
# Clone PyTorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# Environment variables
export USE_CUDA=1
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="12.0"   # For RTX 50xx
export MAX_JOBS=$(nproc)
export CMAKE_PREFIX_PATH=$(python3 -c "from sysconfig import get_paths as gp; print(gp()['data'])")

# Install build dependencies
pip install -r requirements.txt
pip install ninja cmake pyyaml

python3 setup.py clean
python3 setup.py bdist_wheel
pip install dist/torch-*.whl
```

---

## 3. Container Runtime Setup

### Docker

```bash
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

### NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release; echo "${ID}${VERSION_ID}")
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /etc/apt/keyrings/nvidia-container.gpg

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Alternatively, you can use **Podman** and the same NVIDIA container toolkit. The commands are similar (`podman run` instead of `docker run`, etc.).

---

## 4. Dockerfile Example (with FlashInfer + vLLM)

Below is an **example** Dockerfile. You can adapt it for Debian or Ubuntu base images. If you have a separate base image or want a custom PyTorch build inside the container, adjust accordingly.

```dockerfile
# Use the NVIDIA PyTorch image or any suitable base with CUDA 12.8
FROM nvcr.io/nvidia/pytorch:23.08-py3

ENV MAX_JOBS=16
ENV NVCC_THREADS=4
ENV FLASHINFER_ENABLE_AOT=0
ENV USE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST='12.0+PTX'
ENV CCACHE_DIR=/root/.ccache

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    ccache \
    python3-pip \
    python3-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install bitsandbytes

# Build flashinfer
RUN git clone https://github.com/flashinfer-ai/flashinfer.git --recursive /workspace/flashinfer
WORKDIR /workspace/flashinfer
RUN pip3 install -e . -v

# Build vLLM
RUN git clone https://github.com/vllm-project/vllm.git /workspace/vllm
WORKDIR /workspace/vllm
RUN pip3 install --no-cache-dir -r requirements/build.txt
RUN pip3 install --no-cache-dir setuptools_scm
RUN python3 setup.py develop

# (Optional) Start a shell by default
CMD ["bash"]
```

**Build** it:
```bash
mkdir -p ~/vllm/ccache
docker build -t vllm-cu128 -f Dockerfile \
    --build-arg TORCH_CUDA_ARCH_LIST="12.0" \
    --build-arg MAX_JOBS=$(nproc) \
    .
```

*(Adjust Podman usage if desired.)*

---

## 5. Download/Prepare Models (QwQ-32B, WhiteRabbitNeo-13B, DeepSeek, etc.)

```bash
huggingface-cli login
# AWQ or FP4 quantized model examples:
huggingface-cli download Qwen/QwQ-32B-AWQ \
  --local-dir /mnt/models/QwQ-32B-AWQ \
  --local-dir-use-symlinks False
huggingface-cli download TheBloke/WhiteRabbitNeo-13B-AWQ \
  --local-dir /mnt/models/WhiteRabbitNeo-13B-AWQ
...
```

For the sample **QwQ-32B** model:  
- AWQ quantization  
- `--enable-reasoning` and `--tool-call-parser` can be added to the vLLM command for advanced usage.

---

## 6. Run vLLM with AWQ / FP4 Quantization + Reasoning

### QwQ-32B Example (Podman or Docker)

```bash
podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable \
    --net=host --ipc=host \
    -v ~/.cache:/root/.cache \
    --env "HUGGING_FACE_HUB_TOKEN=<your_hf_token>" \
    --env "VLLM_ATTENTION_BACKEND=FLASHINFER" \
    vllm-cu128 \
    vllm serve Qwen/QwQ-32B-AWQ \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-model-len 32768 \
    --disable-sliding-window \
    --generation-config Qwen/QwQ-32B-AWQ \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

### WhiteRabbitNeo-13B Example (Docker command)

```bash
docker run --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 -v /mnt/models:/models \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  vllm-cu128 \
  python3 -m vllm.entrypoints.api_server \
    --model /models/WhiteRabbitNeo-13B-AWQ \
    --quantization awq \
    --dtype half \
    --gpu-memory-utilization 0.90 \
    --cpu-offload-gb 4 \
    --max-model-len 8192
```

### Notes on Reasoning & Tool Usage

- `--enable-reasoning` gives you `reasoning_content` in the assistant’s responses.
- `--enable-auto-tool-choice --tool-call-parser hermes` ensures the model can parse function calls or “tools” automatically.
- If you don’t need this advanced functionality, omit those flags.

---

## 7. Test via HTTP Requests

#### For the OpenAI-compatible endpoint (`api_server`):

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/QwQ-32B-AWQ",
    "prompt": "Explain the concept of black holes.",
    "max_tokens": 512
  }'
```

#### For the vLLM custom endpoint (`vllm serve` CLI):

If using the `vllm serve` command (rather than `api_server`), see the [vLLM docs](https://github.com/vllm-project/vllm) for usage details. Typically, you still get a JSON-based interface on `:8000`.

---

## 8. Memory Tuning

- `--gpu-memory-utilization <float>`: Helps share the GPU with other processes or keep overhead for bigger contexts.
- `--cpu-offload-gb <int>`: Offloads the largest layers to CPU if your GPU memory is insufficient for the entire model.
- `--enable-chunked-prefill`: Reduces peak memory usage for big context windows.
- `--enable-prefix-caching`: Good for repeated chat contexts.

For the **32k** context:
```bash
--max-model-len 32768
--gpu-memory-utilization 0.9
--cpu-offload-gb 6
```

---

## 9. (Optional) Inspect & Save Docker Image

If you want to save your final container image for reuse:

```bash
docker commit <container_id> vllm-cu128:snapshot
docker save -o vllm-cu128_snapshot.tar vllm-cu128:snapshot
```

Then you can `docker load -i vllm-cu128_snapshot.tar` on another machine.

---

## 10. Tips & Troubleshooting

1. **PyTorch Not Found**: Double-check your environment if building from source.
2. **CUDA Errors**: Confirm `nvidia-smi` works; check version mismatch between driver and runtime.
3. **Performance**: Lower `--max-model-len` or use `--gpu-memory-utilization 0.6` if you see OOM issues.
4. **Reasoning Parser**: Make sure the desired parser (`deepseek_r1`, `hermes`, etc.) is recognized by your build version of vLLM if you see errors about unknown parser.

---

Feel free to tailor specific commands to your GPU, environment, or model. For more detailed references, see:

- [vLLM Project GitHub](https://github.com/vllm-project/vllm)
- [FlashInfer Repository](https://github.com/flashinfer-ai/flashinfer)
- [Hugging Face: AWQ Models](https://huggingface.co/search?q=AWQ)

Enjoy your high-performance LLM inference!
