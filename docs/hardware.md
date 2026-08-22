# Hardware and backend guide

Hardware affects speed, memory limits, supported numerical libraries, and the exact measurements you will observe. It should not change the scientific shape of the course: begin with a baseline, record the environment, change one thing, and evaluate fairly.

## What has been verified

The reference course was run on a 32 GB Apple-silicon MacBook Air using Python 3.12 and MLX 0.32.0. The largest recorded Shakespeare scaling candidate used approximately 406 MB of MLX peak accelerator memory, although total process and system memory are larger than this allocator measurement. TinyStories Lesson 1 also verified a 938,496-parameter random model with a 128-token context: the recorded warmed rerun of the bounded-sample audit, provisional tokenizer and inference smoke test took approximately 0.33 seconds and reported 30.7 MB peak MLX allocation. This was not a training benchmark; the first cold invocation took longer.

The repository includes install profiles for platforms currently documented by MLX:

- `apple` — native Apple silicon and macOS;
- `linux-cpu` — Linux CPU backend;
- `linux-cuda12` — Linux with the MLX CUDA 12 backend; and
- `linux-cuda13` — Linux with the MLX CUDA 13 backend.

Run `python3 scripts/system_report.py` before installation for a non-destructive local report.

## Memory tiers

These are planning guides, not guarantees:

| Available memory | Sensible starting scope |
|---|---|
| 8 GB | Complete the early Shakespeare models; monitor memory and reduce batch size before larger work |
| 16 GB | Entire Shakespeare course should be comfortable; use bounded TinyStories subsets and modest model sizes |
| 32 GB | Reference target for all three planned projects, including small-model LoRA experiments |
| 64 GB+ or discrete GPU | More room for larger batches, contexts, pretrained bases, and comparison runs—but preserve small baselines |

On a lower-memory machine, reduce batch size first. Reducing context affects the capability being tested; reducing model width or depth changes architecture. Record those changes rather than comparing the resulting metric as if the recipe were identical.

## Apple silicon

MLX’s [official installation guide](https://ml-explore.github.io/mlx/build/html/install.html) currently requires Apple silicon, native Python 3.10+, and macOS 14+ for the macOS package. This repository pins Python 3.12–3.13 and MLX 0.32.0.

```sh
uv sync --extra apple
uv run --no-sync python scripts/download_tiny_shakespeare.py
```

Apple unified memory is shared by CPU and GPU. The operating system, browser, development server, model arrays, gradients, optimiser state, and cached buffers all use that pool. Do not equate “32 GB unified memory” with 32 GB available exclusively to model weights.

## Linux with NVIDIA CUDA

MLX now documents CUDA backends. At the time this guide was written, its [official requirements](https://ml-explore.github.io/mlx/build/html/install.html#cuda) include Linux with glibc 2.35+, Python 3.10+, NVIDIA architecture SM 7.5+, and backend-specific driver/toolkit requirements. Check the live page before installing because drivers and packages change.

```sh
# Choose exactly one profile after checking the official requirements.
uv sync --extra linux-cuda12
# or
uv sync --extra linux-cuda13
```

The model code should use the MLX default device, but the reference timings and Apple unified-memory discussion will not transfer directly. Record `mx.default_device()`, device information, GPU model, driver, CUDA version, system RAM, and accelerator memory.

This path is documented by the framework but has not yet been clean-clone tested in this repository. Treat initial results as a portability test and report incompatibilities.

## Linux CPU-only

MLX documents a Linux CPU package for glibc 2.35+ and Python 3.10+.

```sh
uv sync --extra linux-cpu
```

Start with bigram, fixed-context, and shortened transformer runs. Full scaling and repeated seeds may take substantially longer than the Apple reference. A slower run is not a worse model; compare prediction metrics separately from elapsed time.

## AMD GPUs

The current MLX installation guide does not provide a ROCm backend. AMD’s [ROCm documentation](https://rocm.docs.amd.com/en/latest/) and PyTorch’s [local installation selector](https://docs.pytorch.org/get-started/locally/) support compatible Linux/PyTorch paths, but this repository’s model scripts use MLX APIs. Running on ROCm therefore requires a deliberate PyTorch port, including equivalent seeding, checkpointing, evaluation, and result schemas.

Do not silently substitute a backend and compare timings as if they were the same experiment. A useful contribution would be a `torch` implementation tested against the MLX logits, parameter counts, data batches, and checkpoint protocol.

## Windows

The current scripts do not claim a native Windows MLX path.

Options are:

1. WSL2 with an NVIDIA GPU that satisfies the Linux MLX CUDA requirements;
2. a remote Linux machine while running the wiki locally; or
3. a PyTorch port using CUDA, CPU, or [PyTorch with DirectML](https://learn.microsoft.com/windows/ai/directml/pytorch-windows).

DirectML makes GPU-backed PyTorch possible on a range of Windows hardware, but it is not a drop-in backend for these MLX source files. Record it as an adaptation project.

## Intel Macs and unsupported systems

The packaged MLX macOS route requires Apple silicon. The wiki still works, but model execution requires a supported Linux environment, remote compute, or a backend port.

## Reduced-compute strategy

When the default recipe is impractical:

1. run the random baseline and one training update first;
2. reduce evaluation batches while labelling the estimate noisier;
3. reduce training steps to verify the full pipeline;
4. reduce batch size to save memory;
5. postpone scaling and multi-seed confirmation until earlier stages work;
6. preserve the original reference result for comparison; and
7. never replace “not run” with a copied metric.

For any altered run, record the exact command and explain why the comparison is not controlled against the original recipe.
