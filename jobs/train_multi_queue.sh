#!/usr/bin/env bash

# Sequential, disconnect-safe queue for the current multi-camera integration.
# It intentionally shares the GPU lock with the single-camera queue.

set -uo pipefail

readonly REPO_ROOT="/workspaces/RotGS"
readonly SESSION_ROOT="/shared/datasets/3D_Scanning_MT"
readonly LOG_ROOT="${REPO_ROOT}/logs/train_multi_queue/session_03"
readonly LOCK_FILE="${REPO_ROOT}/.train_queue_gpu0.lock"
readonly FINAL_ITERATION=30000
readonly RETRY_COUNT=2
readonly RETRY_DELAY_SECONDS=300
readonly MIN_FREE_KIB=$((10 * 1024 * 1024))

readonly SOURCE_PATH="${ROTGS_MULTI_SOURCE:-${SESSION_ROOT}/cleaned/Session_03/Quoellfrisch_top_multi_5d_30d_60d_und_rn_roi_sqr_PNGaR_mixedres}"
readonly OUTPUT_PATH="${ROTGS_MULTI_OUTPUT:-${SESSION_ROOT}/output_RotGS/session_03/Quoellfrisch_top_multi_runs/Quoellfrisch_top_multi_current_calibfixed_30k_wandb}"
readonly MOTION_INITIALIZER="${ROTGS_MULTI_INIT_MOTION:-${SESSION_ROOT}/output_RotGS/session_03/Quoellfrisch_top_multi_runs/Quoellfrisch_top_multi_tiltbound_depthfast_5k}"
readonly WANDB_MODE="${ROTGS_MULTI_WANDB_MODE:-online}"

QUEUE_ATTEMPT=0
QUEUE_FAILURES=0
DELAY_MINUTES=0
DRY_RUN=0

usage() {
    echo "Usage: $0 [--delay-minutes MINUTES] [--dry-run]"
    echo "Override paths with ROTGS_MULTI_SOURCE, ROTGS_MULTI_OUTPUT, and ROTGS_MULTI_INIT_MOTION."
}

while (( $# > 0 )); do
    case "$1" in
        --delay-minutes)
            if (( $# < 2 )); then
                echo "ERROR: --delay-minutes requires a value." >&2
                exit 2
            fi
            DELAY_MINUTES="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "${DELAY_MINUTES}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --delay-minutes must be a non-negative integer." >&2
    exit 2
fi
if [[ ! "${WANDB_MODE}" =~ ^(disabled|offline|online)$ ]]; then
    echo "ERROR: ROTGS_MULTI_WANDB_MODE must be disabled, offline, or online." >&2
    exit 2
fi

mkdir -p "${LOG_ROOT}"
cd "${REPO_ROOT}" || exit 1

timestamp() {
    date --iso-8601=seconds
}

wait_for_gpu() {
    local active_pids
    while true; do
        if ! active_pids="$(
            nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null
        )"; then
            echo "[$(timestamp)] ERROR: could not query GPU state; retrying in 60 seconds."
            sleep 60
            continue
        fi
        if [[ -z "${active_pids//[[:space:]]/}" ]]; then
            return
        fi
        echo "[$(timestamp)] GPU occupied by PID(s): ${active_pids//$'\n'/, }; waiting."
        sleep 60
    done
}

wait_for_disk_space() {
    local output_parent
    local workspace_free
    local output_free
    output_parent="$(dirname "${OUTPUT_PATH}")"
    while true; do
        workspace_free="$(df -Pk "${LOG_ROOT}" | awk 'NR == 2 { print $4 }')"
        output_free="$(df -Pk "${output_parent}" | awk 'NR == 2 { print $4 }')"
        if [[ "${workspace_free}" =~ ^[0-9]+$             && "${output_free}" =~ ^[0-9]+$             && "${workspace_free}" -ge "${MIN_FREE_KIB}"             && "${output_free}" -ge "${MIN_FREE_KIB}" ]]; then
            return
        fi
        echo "[$(timestamp)] Insufficient disk space; waiting five minutes."
        sleep 300
    done
}

run_multi_job() {
    local final_model="${OUTPUT_PATH}/point_cloud/iteration_${FINAL_ITERATION}/point_cloud.ply"
    local log_suffix=""
    local log_path
    local exit_code
    local -a command=(
        uv run python train_multi.py
        -s "${SOURCE_PATH}"
        --name "${OUTPUT_PATH}"
        --iterations "${FINAL_ITERATION}"
        --init_motion "${MOTION_INITIALIZER}"
        --freeze_motion
        --bootstrap_iterations 5000
        --pose_warmup_iterations 0
        --appearance_warmup_iterations 0
        --densify_until_iter 20000
        --max_gaussians 500000
        --rotation_direction 1
        --axis_mode bounded_tilt
        --axis_tilt_deviation_limit_deg 5
        --max_residual_angle_deg 1
        --max_sweep_error_deg 4
        --max_phase_offset_deg 30
        --lambda_dssim 0.2
        --lambda_foreground_rgb 1.0
        --lambda_full_rgb 0.1
        --lambda_alpha 0.1
        --lambda_silhouette 0.2
        --wo_tiny
        --wo_flow
        --test_iterations 5000 10000 15000 20000 25000 30000
        --save_iterations 5000 10000 15000 20000 25000 30000
        --checkpoint_iterations 5000 10000 15000 20000 25000 30000
        --wandb_mode "${WANDB_MODE}"
        --wandb_run_name "quoellfrisch-top-multi-current-calibfixed-30k"
        --wandb_group "multi-camera-integration"
        --wandb_tags multi-camera rigid fresh-geometry calibration-fixed
        --wandb_artifacts best_and_final
    )

    if [[ -f "${final_model}" && "${ROTGS_RERUN_COMPLETED:-0}" != "1" ]]; then
        echo "[$(timestamp)] SKIP: final model already exists at ${final_model}"
        return 0
    fi

    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'

    if (( DRY_RUN )); then
        return 0
    fi

    if (( QUEUE_ATTEMPT > 0 )); then
        log_suffix=".retry-${QUEUE_ATTEMPT}"
    fi
    log_path="${LOG_ROOT}/quoellfrisch-top-multi-current${log_suffix}.log"

    wait_for_disk_space
    wait_for_gpu
    echo "[$(timestamp)] START multi-camera run (attempt $((QUEUE_ATTEMPT + 1)))"
    if "${command[@]}" >"${log_path}" 2>&1; then
        exit_code=0
    else
        exit_code=$?
    fi

    if (( exit_code == 0 )) && [[ -f "${final_model}" ]]; then
        echo "[$(timestamp)] DONE multi-camera run"
    else
        QUEUE_FAILURES=$((QUEUE_FAILURES + 1))
        echo "[$(timestamp)] FAILED multi-camera run (exit ${exit_code}); log: ${log_path}"
    fi
}

if (( ! DRY_RUN )); then
    command -v flock >/dev/null 2>&1 || {
        echo "ERROR: flock is required." >&2
        exit 1
    }
    command -v nvidia-smi >/dev/null 2>&1 || {
        echo "ERROR: nvidia-smi is required." >&2
        exit 1
    }
    exec 9>"${LOCK_FILE}"
    if ! flock -n 9; then
        echo "ERROR: another RotGS queue is already using GPU 0." >&2
        exit 1
    fi
fi

if (( DELAY_MINUTES > 0 && ! DRY_RUN )); then
    echo "[$(timestamp)] Waiting ${DELAY_MINUTES} minute(s) before starting."
    sleep "$((DELAY_MINUTES * 60))"
fi

for ((attempt = 0; attempt <= RETRY_COUNT; attempt++)); do
    QUEUE_ATTEMPT="${attempt}"
    QUEUE_FAILURES=0
    run_multi_job
    if (( DRY_RUN || QUEUE_FAILURES == 0 )); then
        break
    fi
    if (( attempt < RETRY_COUNT )); then
        echo "[$(timestamp)] Waiting before retry."
        sleep "${RETRY_DELAY_SECONDS}"
    fi
done

if (( QUEUE_FAILURES > 0 )); then
    exit 1
fi
