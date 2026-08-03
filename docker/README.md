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

# Stamp a real version (else anuga.__version__ is 0.0.0+unknown, since .git is
# excluded from the build context):
docker build -f docker/Dockerfile.gpu \
  --build-arg ANUGA_VERSION="$(python _git_version.py)" -t anuga:gpu .
```

### Published (pre-release) image

The GPU image is built from the **develop** branch (v4.0 line), which isn't
released yet — so it's published to GHCR under a **pre-release** tag, not
`latest`:

```bash
docker pull ghcr.io/anuga-community/anuga:develop-gpu     # branch channel
# or an immutable commit:  ghcr.io/anuga-community/anuga:sha-<short>-gpu
```

Publishing a container from develop is fine — it's an artifact, not a source
release. The `docker-publish.yml` workflow (manual `workflow_dispatch`,
`build_gpu=true`) tags `develop-gpu` + `sha-<short>-gpu`; version tags + `latest`
only appear on a GitHub Release. Until CI has a big enough runner for the ~15 GB
NVHPC base, the reliable path is to **build locally and push**:

```bash
docker build -f docker/Dockerfile.gpu \
  --build-arg ANUGA_VERSION="$(python _git_version.py)" \
  -t ghcr.io/anuga-community/anuga:develop-gpu .
echo "$GHCR_PAT" | docker login ghcr.io -u <user> --password-stdin
docker push ghcr.io/anuga-community/anuga:develop-gpu
```

**Make the package public** (so AWS pulls without credentials). Two steps, both
require an org **owner**:
1. Allow public packages org-wide: **Org Settings → "Code, planning, and
   automation" → Packages** → enable public packages.
2. On the package: **Package settings → Danger Zone → Change visibility → Public**
   (also link it to `anuga_core` via *Manage Actions access* so the workflow can
   update it).

`ghcr.io/anuga-community/anuga:develop-gpu` is currently **published and public**
(anonymous `docker pull` verified).

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
| `INPUT_S3` | `aws s3 sync $INPUT_S3 /work` before the run — input lands **in place** (so `python run.py` finds `run.py` at `/work/run.py`) |
| `OUTPUT_S3` | `aws s3 sync <ANUGA_OUTPUT_DIR> $OUTPUT_S3` after a successful run |
| `ANUGA_OUTPUT_DIR` | what to upload (default `/work` — captures output wherever the script writes it, e.g. `MODEL_OUTPUTS/`; set to a subdir to skip re-sending large inputs) |
| `STAGE_OUT_ON_FAILURE=1` | also upload when the run fails |
| `AWS_BATCH_JOB_ARRAY_INDEX` | appended to `OUTPUT_S3` (array jobs land in `.../0/`, `.../1/`, …) |

Credentials come from the environment / instance role — none are baked in. Test
the flow locally with a mounted dir and no S3:

```bash
docker run --rm -v "$PWD/docker/data:/work" anuga:cpu python /work/my_run.py
# results appear in docker/data/output/
```

---

## Simplest AWS path — one self-terminating GPU instance

For a **single** GPU run with **no standing infrastructure** (ideal if you have
no local GPU), `aws_run_gpu.sh` launches one GPU EC2 instance that pulls the
image, runs your command with S3 in/out, uploads results, and **terminates
itself**. You pay only for the job's instance-hours.

```bash
# upload your project (run script + data), launch, walk away:
docker/aws_run_gpu.sh \
  --upload  ./my_tohoku_project \
  --input   s3://my-bucket/anuga/in \
  --output  s3://my-bucket/anuga/out \
  --command "python run_Tohoku.py -alg DE0"
# results (+ anuga-run.log) appear under --output; the instance is gone.
```

Validate first (checks creds, GPU quota, AMI; prints the plan; launches nothing):
```bash
docker/aws_run_gpu.sh --region ap-southeast-2 \
  --output s3://my-bucket/anuga/out --command "python run.py" --dry-run
```

Defaults to `g5.2xlarge` (1× A10G / cc86, 8 vCPU, 32 GB — a good balance since
mesh-gen + DEM fitting are CPU-bound). Flags: `--dry-run`, `--instance`, `--spot`,
`--keep` (don't self-terminate, for debugging), `--ami`, `--instance-profile`,
`--region`, `--disk`. Prereqs: `awscli` configured, an S3 bucket, and a one-time
**GPU vCPU service-quota increase** (new accounts start at 0 for G/P families).
Each user runs this in **their own AWS account and pays for their own usage**.

**New to AWS?** Follow [`AWS_SETUP.md`](AWS_SETUP.md) — account, budget alerts,
the quota increase, region + bucket, then dry-run and launch.

A ~1M-triangle single-GPU run fits comfortably (only a few GB of GPU memory).

---

## Running many jobs on AWS Batch

For sweeps/ensembles, push to a registry AWS can pull (GHCR public, or your
account's ECR) and use Batch array jobs:

```bash
# ECR example
aws ecr create-repository --repository-name anuga || true
aws ecr get-login-password | docker login --username AWS --password-stdin \
  "$AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com"
docker tag anuga:gpu "$AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/anuga:gpu"
docker push "$AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/anuga:gpu"
```

### AWS Batch (GPU)

1. **Compute environment** — EC2 with GPU instance types (`g4dn`/`g5`/`g6`/`p4d`/`p5`;
   `p3`/V100 is not covered by the default image — see GPU_ARCH note in Dockerfile.gpu)
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

- **File ownership / run as your host user.** By default the container runs as
  **root**, so anything it writes to the bind-mounted dir (e.g. `MODEL_OUTPUTS/*.sww`)
  is owned by `root` on the host. A later non-root run then can't overwrite those
  files, and ANUGA silently reopens+appends to the stale `.sww` (garbled time
  axis) instead of truncating. Avoid it by running as yourself:
  ```bash
  docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -v "$PWD:/work" anuga:gpu python run.py
  ```
  (The images set `HOME=/tmp` so `--user` runs have a writable config dir. With
  compose, `export UID GID` first — see the header of docker-compose.yml.)
- **Image size:** the GPU image's NVHPC base is ~10–15 GB. A slimmer multi-stage
  runtime (CUDA-runtime base + copied NVHPC redistributable libs + the venv) is
  drafted in `Dockerfile.gpu.slim` (**experimental** — needs local build-testing
  to confirm the runtime-lib set). It should drop the image to a few GB, cutting
  storage and AWS cold-start pull time.
- **Version string:** `.git` is excluded from the build context, so a
  source-built GPU image reports `0.0.0+unknown` for `anuga.__version__`
  (cosmetic; the code is the checkout's).
- **`ANUGA_DEFAULT_COMPUTE_MODE=unified`** turns on the mode-2 path. On the GPU
  image with a visible GPU it offloads; without `--gpus` it falls back to the
  (slow) host path. Leave it unset for legacy CPU runs.
