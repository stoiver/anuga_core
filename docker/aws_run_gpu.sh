#!/usr/bin/env bash
#
# aws_run_gpu.sh — run a single ANUGA GPU job on AWS with no standing infra.
#
# Launches ONE GPU EC2 instance that pulls the anuga:gpu container, runs your
# command with S3 input/output, uploads results, and TERMINATES itself. You pay
# only for the instance-hours the job uses (it self-destructs when done).
#
# Prereqs: awscli configured (`aws configure`); an S3 bucket you can read/write;
# a one-time GPU vCPU service-quota increase for the instance family (new
# accounts start at 0 for G/P instances).
#
# Typical use — upload your project (run script + data), then launch:
#   ./aws_run_gpu.sh \
#       --upload   ./my_tohoku_project \
#       --input    s3://my-bucket/anuga/in \
#       --output   s3://my-bucket/anuga/out \
#       --command  "python run_Tohoku.py -alg DE0"
#
# Results (and anuga-run.log) appear under --output. The instance is gone.
#
# Add --dry-run to validate creds / GPU quota / AMI and print the launch plan
# without creating or charging anything.
#
# NOTE: authored but not yet exercised against a live account — test with your
# own creds and a tiny run first.
set -euo pipefail

# ---- defaults ---------------------------------------------------------------
IMAGE="ghcr.io/anuga-community/anuga:develop-gpu"   # pre-release (built from develop)
INSTANCE_TYPE="g5.2xlarge"        # 1x A10G (24GB, cc86), 8 vCPU, 32GB RAM
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
DISK_GB=100                       # root volume; DEMs + SWW output can be large
COMPUTE_MODE="unified"            # GPU offload for mode-2 domains
INPUT_S3="" ; OUTPUT_S3="" ; COMMAND="" ; UPLOAD_DIR="" ; OUTPUT_DIR=""
INSTANCE_PROFILE="" ; AMI="" ; SSH_KEY="" ; SPOT=0 ; KEEP=0 ; DRY=0

usage() {
  sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

# ---- args -------------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --image)            IMAGE="$2"; shift 2 ;;
    --instance)         INSTANCE_TYPE="$2"; shift 2 ;;
    --region)           REGION="$2"; shift 2 ;;
    --disk)             DISK_GB="$2"; shift 2 ;;
    --input)            INPUT_S3="$2"; shift 2 ;;
    --output)           OUTPUT_S3="$2"; shift 2 ;;
    --command)          COMMAND="$2"; shift 2 ;;
    --upload)           UPLOAD_DIR="$2"; shift 2 ;;
    --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;   # ANUGA_OUTPUT_DIR in container
    --compute-mode)     COMPUTE_MODE="$2"; shift 2 ;;
    --instance-profile) INSTANCE_PROFILE="$2"; shift 2 ;;
    --ami)              AMI="$2"; shift 2 ;;
    --ssh-key)          SSH_KEY="$2"; shift 2 ;;
    --spot)             SPOT=1; shift ;;
    --keep)             KEEP=1; shift ;;              # don't self-terminate (debug)
    --dry-run)          DRY=1; shift ;;              # validate + print plan; launch nothing
    -h|--help)          usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

die() { echo "ERROR: $*" >&2; exit 1; }
[ -n "$OUTPUT_S3" ] || die "--output s3://... is required"
[ -n "$COMMAND" ]   || die "--command '...' is required"
[ -n "$REGION" ]    || die "no region (pass --region or set AWS_REGION)"
[ -n "$UPLOAD_DIR" ] && [ -z "$INPUT_S3" ] && die "--upload needs --input to upload to"
AWS=(aws --region "$REGION")

bucket_of() { echo "$1" | sed -E 's#^s3://([^/]+).*#\1#'; }

# ---- optional: upload a local project dir to the input prefix ---------------
if [ -n "$UPLOAD_DIR" ]; then
  if [ "$DRY" = "1" ]; then
    echo "[dry-run] would upload ${UPLOAD_DIR} -> ${INPUT_S3}"
  else
    echo ">> uploading ${UPLOAD_DIR} -> ${INPUT_S3}"
    aws s3 sync "$UPLOAD_DIR" "$INPUT_S3"
  fi
fi

# ---- resolve a GPU-ready AMI (Docker + NVIDIA toolkit + driver preinstalled) -
if [ -z "$AMI" ]; then
  echo ">> resolving Deep Learning Base GPU AMI via SSM"
  AMI=$("${AWS[@]}" ssm get-parameter \
    --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
    --query 'Parameter.Value' --output text 2>/dev/null) \
    || die "could not resolve the DLAMI AMI id via SSM; pass --ami ami-xxxxx (a GPU AMI with Docker + the NVIDIA Container Toolkit)"
fi
echo ">> AMI: $AMI   instance: $INSTANCE_TYPE   region: $REGION"

# ---- ensure an instance profile with S3 access ------------------------------
if [ -z "$INSTANCE_PROFILE" ]; then
  INSTANCE_PROFILE="anuga-gpu-runner"
  ROLE="$INSTANCE_PROFILE"
  if aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" >/dev/null 2>&1; then
    : # already exists
  elif [ "$DRY" = "1" ]; then
    echo "[dry-run] would create IAM instance profile '$INSTANCE_PROFILE' (S3 access to the input/output buckets)"
  else
    echo ">> creating IAM instance profile '$INSTANCE_PROFILE' (S3 access)"
    IN_B=$(bucket_of "${INPUT_S3:-$OUTPUT_S3}") ; OUT_B=$(bucket_of "$OUTPUT_S3")
    aws iam create-role --role-name "$ROLE" \
      --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
    aws iam put-role-policy --role-name "$ROLE" --policy-name s3-access \
      --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\",\"s3:ListBucket\"],\"Resource\":[\"arn:aws:s3:::${IN_B}\",\"arn:aws:s3:::${IN_B}/*\",\"arn:aws:s3:::${OUT_B}\",\"arn:aws:s3:::${OUT_B}/*\"]}]}" >/dev/null
    aws iam create-instance-profile --instance-profile-name "$INSTANCE_PROFILE" >/dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "$INSTANCE_PROFILE" --role-name "$ROLE" >/dev/null
    echo ">> waiting for IAM propagation..."; sleep 15
  fi
fi

# ---- build user-data (runs at boot on the GPU instance) ---------------------
TERMINATE="shutdown -h now"
[ "$KEEP" = "1" ] && TERMINATE="echo '[--keep] leaving instance up; terminate it manually when done.'"
USER_DATA=$(cat <<EOF
#!/bin/bash
set -x
exec > /var/log/anuga-run.log 2>&1
docker pull "${IMAGE}" || true
docker run --rm --gpus all \
  -e ANUGA_DEFAULT_COMPUTE_MODE='${COMPUTE_MODE}' \
  ${INPUT_S3:+-e INPUT_S3='${INPUT_S3}'} \
  -e OUTPUT_S3='${OUTPUT_S3}' \
  ${OUTPUT_DIR:+-e ANUGA_OUTPUT_DIR='${OUTPUT_DIR}'} \
  "${IMAGE}" sh -c '${COMMAND}'
rc=\$?
aws s3 cp /var/log/anuga-run.log "${OUTPUT_S3%/}/anuga-run.log" || true
echo "anuga job exited rc=\$rc"
${TERMINATE}
EOF
)

# ---- launch -----------------------------------------------------------------
RUN_ARGS=(
  --image-id "$AMI"
  --instance-type "$INSTANCE_TYPE"
  --count 1
  --iam-instance-profile "Name=$INSTANCE_PROFILE"
  --instance-initiated-shutdown-behavior terminate
  --metadata-options "HttpEndpoint=enabled,HttpTokens=required,HttpPutResponseHopLimit=2"
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${DISK_GB},VolumeType=gp3}"
  --user-data "$USER_DATA"
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=anuga-gpu-run}]"
)
[ -n "$SSH_KEY" ] && RUN_ARGS+=(--key-name "$SSH_KEY")
[ "$SPOT" = "1" ] && RUN_ARGS+=(--instance-market-options "MarketType=spot")
# The metadata hop limit of 2 is REQUIRED so the container can reach IMDSv2 for
# the instance-profile credentials the entrypoint's aws sync uses.

if [ "$DRY" = "1" ]; then
  echo
  echo "================= DRY RUN — nothing launched, nothing charged ================="
  echo "-- identity --"
  aws sts get-caller-identity --output text 2>&1 | sed 's/^/  /' || echo "  (could not verify creds — is the CLI configured?)"
  echo "-- GPU on-demand vCPU quota (needs >= this instance's vCPUs) --"
  q=$("${AWS[@]}" service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA \
        --query 'Quota.Value' --output text 2>/dev/null) \
    && echo "  current 'Running On-Demand G and P instances' = ${q} vCPUs" \
    || echo "  (could not read quota — check Service Quotas; new accounts start at 0, see AWS_SETUP.md step 6)"
  echo "-- plan --"
  printf '  %-16s %s\n' image "$IMAGE" instance "$INSTANCE_TYPE (spot=$SPOT, disk=${DISK_GB}GB)" \
    region "$REGION" ami "$AMI" profile "$INSTANCE_PROFILE" \
    input "${INPUT_S3:-<none>}" output "$OUTPUT_S3" command "$COMMAND" compute-mode "$COMPUTE_MODE"
  echo "-- ec2 run-instances permission check --"
  if aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" >/dev/null 2>&1; then
    "${AWS[@]}" ec2 run-instances "${RUN_ARGS[@]}" --dry-run 2>&1 | sed 's/^/  /' | head -3
  else
    echo "  (skipped — instance profile not created yet; it would be created on a real run)"
  fi
  echo "==============================================================================="
  echo "Re-run without --dry-run to launch."
  exit 0
fi

echo ">> launching..."
IID=$("${AWS[@]}" ec2 run-instances "${RUN_ARGS[@]}" --query 'Instances[0].InstanceId' --output text)
echo ">> instance $IID launched ($INSTANCE_TYPE${SPOT:+, spot})."
echo ">> it will run: ${COMMAND}"
echo ">> results ->   ${OUTPUT_S3}    (+ anuga-run.log)"
if [ "$KEEP" = "1" ]; then
  echo ">> --keep set: instance will NOT self-terminate. Terminate with:"
  echo "     aws --region $REGION ec2 terminate-instances --instance-ids $IID"
else
  echo ">> instance self-terminates when the job finishes. Watch it:"
  echo "     aws --region $REGION ec2 describe-instances --instance-ids $IID --query 'Reservations[0].Instances[0].State.Name' --output text"
fi
