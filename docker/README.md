# ANUGA in Docker (local + AWS)

Two images:

| Image | Base | ANUGA source | Use |
|-------|------|--------------|-----|
| `anuga:cpu` | `python:3.12-slim` | PyPI wheel | CPU runs; small, fast; CPU AWS Batch |
| `anuga:gpu` | `nvcr.io/nvidia/nvhpc:*-devel` | built from this checkout with `nvc` | GPU offload; local Blackwell laptop **and** AWS GPU instances |

Both share `anuga-entrypoint.sh`: a headless entrypoint that optionally syncs
input from S3, runs the command you pass, and syncs `output/` back to S3 —
ideal for AWS Batch, and a no-op locally when `INPUT_S3`/`OUTPUT_S3` are unset.

There are **no GPU wheels** (the offload extension is OpenMP-target code only
`nvc` compiles), so the GPU image must build from source.

---

## Prerequisites

- **Docker** (`docker --version`).
- **GPU image only:** NVIDIA driver on the host + the **NVIDIA Container
  Toolkit**, then `docker run --gpus all ... nvidia-smi` should list your GPU.

---

## Build & test locally

Put data in `docker/data/` (mounted at `/work` by compose).

### CPU

```bash
docker build -f docker/Dockerfile.cpu -t anuga:cpu .
docker run --rm anuga:cpu python -c "import anuga; print(anuga.__version__)"
```

### GPU

First confirm the NVHPC base tag exists (and supports your arch — Blackwell
`cc120` needs NVHPC ≥ 25.1). Tags: <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nvhpc/tags>

```bash
docker pull nvcr.io/nvidia/nvhpc:25.9-devel-cuda_multi-ubuntu24.04   # adjust if 404

# Multi-arch (AWS + laptop). Large base + fat binary => slow, one-time.
docker build -f docker/Dockerfile.gpu -t anuga:gpu .

# ...or a faster laptop-only build (RTX 50-series = cc120):
docker build -f docker/Dockerfile.gpu \
  --build-arg GPU_ARCH=cc120 -t anuga:gpu-local .
```

Verify offload actually reaches the GPU (needs `--gpus all`):

```bash
docker run --rm --gpus all anuga:gpu python - <<'PY'
import anuga
print("supported:", anuga.gpu_offload_supported(), "enabled:", anuga.gpu_offload_enabled())
PY
```

`supported: True` means the GPU is visible and the offload build works. A small
unified run should print the "GPU Domain" startup banner:

```bash
docker run --rm --gpus all -e ANUGA_DEFAULT_COMPUTE_MODE=unified \
  -v "$PWD/docker/data:/work" anuga:gpu python /work/my_run.py -alg DE0
```

### Via compose

```bash
docker compose -f docker/docker-compose.yml build anuga-gpu
docker compose -f docker/docker-compose.yml run --rm anuga-gpu python /work/my_run.py
```

---

## S3 entrypoint (the AWS batch pattern)

The entrypoint runs whatever command you give it from `$ANUGA_WORKDIR` (`/work`)
and, driven by env vars:

| Env var | Effect |
|---------|--------|
| `INPUT_S3` | `aws s3 cp $INPUT_S3 /work/input --recursive` before the run |
| `OUTPUT_S3` | `aws s3 cp /work/output $OUTPUT_S3 --recursive` after a successful run |
| `STAGE_OUT_ON_FAILURE=1` | also upload output when the run fails |
| `AWS_BATCH_JOB_ARRAY_INDEX` | appended to `OUTPUT_S3` (array jobs land in `.../0/`, `.../1/`, …) |

Write results to `/work/output`. Credentials come from the environment / instance
role — none are baked in. Test the flow locally with a mounted dir and no S3:

```bash
docker run --rm -v "$PWD/docker/data:/work" anuga:cpu python /work/my_run.py
# results appear in docker/data/output/
```

---

## Running on AWS

Push to a registry AWS can pull (GHCR public, or your account's ECR):

```bash
# ECR example
aws ecr create-repository --repository-name anuga || true
aws ecr get-login-password | docker login --username AWS --password-stdin \
  "$AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com"
docker tag anuga:gpu "$AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/anuga:gpu"
docker push "$AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/anuga:gpu"
```

### AWS Batch (GPU)

1. **Compute environment** — EC2 with GPU instance types (`p3`/`g4dn`/`g5`/`p4d`/`p5`)
   using the **ECS GPU-optimized AMI** (driver + `nvidia-container-runtime`
   preinstalled). Spot is fine for interruptible runs.
2. **Job definition** — key fields:
   ```json
   {
     "image": "<registry>/anuga:gpu",
     "resourceRequirements": [
       {"type": "GPU",    "value": "1"},
       {"type": "VCPU",   "value": "8"},
       {"type": "MEMORY", "value": "32768"}
     ],
     "environment": [
       {"name": "INPUT_S3",  "value": "s3://my-bucket/tohoku/input/"},
       {"name": "OUTPUT_S3", "value": "s3://my-bucket/tohoku/results/"},
       {"name": "ANUGA_DEFAULT_COMPUTE_MODE", "value": "unified"}
     ],
     "command": ["python", "/work/input/my_run.py", "-alg", "DE0"],
     "jobRoleArn": "arn:aws:iam::<acct>:role/anuga-batch-s3"
   }
   ```
   Requesting `GPU: 1` makes ECS use the NVIDIA runtime and set
   `NVIDIA_VISIBLE_DEVICES` — **do not** pass `--gpus`. The `jobRoleArn` grants
   the S3 access the entrypoint uses.
3. **Submit** — a single job, or an **array job** (`arraySize: N`) for a sweep;
   each index gets its own `OUTPUT_S3/<index>/` prefix automatically.

> Multi-GPU MPI (e.g. 8×A100 on a `p4d`) needs #ranks == #visible GPUs — see the
> project note on MPI mode-2. Use a multi-node-parallel Batch job for cross-node MPI.

---

## Notes & caveats

- **Image size:** the GPU image's NVHPC base is ~10–15 GB. A slimmer multi-stage
  runtime (CUDA-runtime base + copied NVHPC redistributable libs + the venv) is a
  worthwhile follow-up once the single-stage image is confirmed working.
- **Version string:** `.git` is excluded from the build context, so a
  source-built GPU image reports `0.0.0+unknown` for `anuga.__version__`
  (cosmetic; the code is the checkout's).
- **`ANUGA_DEFAULT_COMPUTE_MODE=unified`** turns on the mode-2 path. On the GPU
  image with a visible GPU it offloads; without `--gpus` it falls back to the
  (slow) host path. Leave it unset for legacy CPU runs.
