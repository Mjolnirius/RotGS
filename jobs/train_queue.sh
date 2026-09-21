#!/usr/bin/env bash

# Sequential, disconnect-safe RotGS queue for the single-camera Session_03
# datasets. Launch this script with nohup; it deliberately runs one GPU job at
# a time because train.py uses CUDA device 0.

set -uo pipefail

readonly REPO_ROOT="/workspaces/RotGS"
readonly LOG_ROOT="${REPO_ROOT}/logs/train_queue/session_03"
readonly LOCK_FILE="${REPO_ROOT}/.train_queue_gpu0.lock"
readonly FINAL_ITERATION=30000
readonly RETRY_COUNT=2
readonly RETRY_DELAY_SECONDS=300
readonly MIN_FREE_KIB=$((10 * 1024 * 1024))

QUEUE_ATTEMPT=0
QUEUE_FAILURES=0
DELAY_MINUTES=0

usage() {
    echo "Usage: $0 [--delay-minutes MINUTES]"
}

while (( $# > 0 )); do
    case "$1" in
        --delay-minutes)
            if (( $# < 2 )); then
                echo "ERROR: --delay-minutes requires a value." >&2
                usage >&2
                exit 2
            fi
            DELAY_MINUTES="$2"
            shift 2
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
readonly DELAY_MINUTES

mkdir -p "${LOG_ROOT}"
cd "${REPO_ROOT}" || exit 1

if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock is required but is not installed." >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is required but is not installed." >&2
    exit 1
fi

# Prevent two copies of this queue from running simultaneously.
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "ERROR: another RotGS queue is already running." >&2
    exit 1
fi

if (( DELAY_MINUTES > 0 )); then
    echo "[$(date --iso-8601=seconds)] Queue scheduled; waiting ${DELAY_MINUTES} minute(s)."
    sleep "$((DELAY_MINUTES * 60))"
    echo "[$(date --iso-8601=seconds)] Delay finished; starting queue."
fi

timestamp() {
    date --iso-8601=seconds
}

wait_for_gpu() {
    local active_pids
    while true; do
        if ! active_pids="$(
            nvidia-smi \
                --query-compute-apps=pid \
                --format=csv,noheader,nounits 2>/dev/null
        )"; then
            echo "[$(timestamp)] ERROR: could not query GPU state; retrying in 60 seconds."
            sleep 60
            continue
        fi

        if [[ -z "${active_pids//[[:space:]]/}" ]]; then
            return
        fi

        echo "[$(timestamp)] GPU is occupied by PID(s): ${active_pids//$'\n'/, }; waiting 60 seconds."
        sleep 60
    done
}

wait_for_disk_space() {
    local output_path="$1"
    local output_parent
    local workspace_free
    local output_free

    output_parent="$(dirname "${output_path}")"
    while true; do
        workspace_free="$(df -Pk "${LOG_ROOT}" | awk 'NR == 2 { print $4 }')"
        output_free="$(df -Pk "${output_parent}" | awk 'NR == 2 { print $4 }')"

        if [[ "${workspace_free}" =~ ^[0-9]+$ \
            && "${output_free}" =~ ^[0-9]+$ \
            && "${workspace_free}" -ge "${MIN_FREE_KIB}" \
            && "${output_free}" -ge "${MIN_FREE_KIB}" ]]; then
            return
        fi

        echo "[$(timestamp)] Insufficient disk space for the next run " \
            "(workspace=${workspace_free:-unknown} KiB, output=${output_free:-unknown} KiB); " \
            "waiting 5 minutes."
        sleep 300
    done
}

run_job() {
    local label="$1"
    local output_path="$2"
    shift 2

    local final_model="${output_path}/point_cloud/iteration_${FINAL_ITERATION}/point_cloud.ply"
    local log_suffix=""
    local log_path
    local exit_code
    local argument_index
    local -a command=("$@")

    if [[ -f "${final_model}" ]] \
        && { (( QUEUE_ATTEMPT > 0 )) || [[ "${ROTGS_RERUN_COMPLETED:-0}" != "1" ]]; }; then
        echo "[$(timestamp)] SKIP ${label}: final model already exists at ${final_model}"
        return 0
    fi

    if (( QUEUE_ATTEMPT > 0 )); then
        log_suffix=".retry-${QUEUE_ATTEMPT}"
        for ((argument_index = 0; argument_index < ${#command[@]} - 1; argument_index++)); do
            if [[ "${command[argument_index]}" == "--wandb_run_name" ]]; then
                command[argument_index + 1]="${command[argument_index + 1]}-retry-${QUEUE_ATTEMPT}"
                break
            fi
        done
    fi
    log_path="${LOG_ROOT}/${label}${log_suffix}.log"

    wait_for_disk_space "${output_path}"
    wait_for_gpu

    echo "[$(timestamp)] START ${label} (attempt $((QUEUE_ATTEMPT + 1)))"
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\nLog: %s\n' "${log_path}"

    if "${command[@]}" >"${log_path}" 2>&1; then
        exit_code=0
    else
        exit_code=$?
    fi

    if (( exit_code == 0 )) && [[ -f "${final_model}" ]]; then
        echo "[$(timestamp)] DONE ${label} (attempt $((QUEUE_ATTEMPT + 1)))"
    else
        QUEUE_FAILURES=$((QUEUE_FAILURES + 1))
        if (( exit_code == 0 )); then
            echo "[$(timestamp)] FAILED ${label}: command exited successfully but the final model is missing; continuing."
        else
            echo "[$(timestamp)] FAILED ${label} (exit ${exit_code}); continuing with the next dataset."
        fi
    fi
}

# Calibration_03B, outdated_versions, and the *_multi_* dataset are omitted.

run_all_jobs() {

run_job \
    "chips-blau-down-s03-densify13k" \
    "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Chips_blau_down_und_PNGaR_3029_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    uv run python train.py \
    -s "/shared/datasets/3D_Scanning_MT/cleaned/Session_03/Chips_blau_down_und_rn_roi_sqr_PNGaR_3029" \
    --name "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Chips_blau_down_und_PNGaR_3029_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    --iterations 30000 \
    --densify_until_iter 13000 \
    --rotation_direction 1 \
    --max_sweep_error_deg 4 \
    --axis_mode bounded_tilt \
    --fixed_camera \
    --random \
    --wo_tiny \
    --wo_flow \
    --test_iterations 5000 10000 15000 20000 25000 30000 \
    --save_iterations 5000 10000 15000 20000 25000 30000 \
    --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
    --wandb_mode online \
    --wandb_run_name "chips-blau-down-s03-densify13k"

run_job \
    "chips-blau-s03-densify13k" \
    "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Chips_blau_und_PNGaR_3029_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    uv run python train.py \
    -s "/shared/datasets/3D_Scanning_MT/cleaned/Session_03/Chips_blau_und_rn_roi_sqr_PNGaR_3029" \
    --name "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Chips_blau_und_PNGaR_3029_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    --iterations 30000 \
    --densify_until_iter 13000 \
    --rotation_direction 1 \
    --max_sweep_error_deg 4 \
    --axis_mode bounded_tilt \
    --fixed_camera \
    --random \
    --wo_tiny \
    --wo_flow \
    --test_iterations 5000 10000 15000 20000 25000 30000 \
    --save_iterations 5000 10000 15000 20000 25000 30000 \
    --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
    --wandb_mode online \
    --wandb_run_name "chips-blau-s03-densify13k"

run_job \
    "kaegifret-down-s03-densify13k" \
    "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Kaegifret_down_und_PNGaR_2226_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    uv run python train.py \
    -s "/shared/datasets/3D_Scanning_MT/cleaned/Session_03/Kaegifret_down_und_rn_roi_sqr_PNGaR_2226" \
    --name "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Kaegifret_down_und_PNGaR_2226_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    --iterations 30000 \
    --densify_until_iter 13000 \
    --rotation_direction 1 \
    --max_sweep_error_deg 4 \
    --axis_mode bounded_tilt \
    --fixed_camera \
    --random \
    --wo_tiny \
    --wo_flow \
    --test_iterations 5000 10000 15000 20000 25000 30000 \
    --save_iterations 5000 10000 15000 20000 25000 30000 \
    --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
    --wandb_mode online \
    --wandb_run_name "kaegifret-down-s03-densify13k"

run_job \
    "kaegifret-top-s03-densify13k" \
    "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Kaegifret_top_und_PNGaR_2226_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    uv run python train.py \
    -s "/shared/datasets/3D_Scanning_MT/cleaned/Session_03/Kaegifret_top_und_rn_roi_sqr_PNGaR_2226" \
    --name "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Kaegifret_top_und_PNGaR_2226_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    --iterations 30000 \
    --densify_until_iter 13000 \
    --rotation_direction 1 \
    --max_sweep_error_deg 4 \
    --axis_mode bounded_tilt \
    --fixed_camera \
    --random \
    --wo_tiny \
    --wo_flow \
    --test_iterations 5000 10000 15000 20000 25000 30000 \
    --save_iterations 5000 10000 15000 20000 25000 30000 \
    --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
    --wandb_mode online \
    --wandb_run_name "kaegifret-top-s03-densify13k"

run_job \
    "leermond-s03-densify13k" \
    "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Leermond_und_PNGaR_2871_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    uv run python train.py \
    -s "/shared/datasets/3D_Scanning_MT/cleaned/Session_03/Leermond_und_rn_roi_sqr_PNGaR_2871" \
    --name "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Leermond_und_PNGaR_2871_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    --iterations 30000 \
    --densify_until_iter 13000 \
    --rotation_direction 1 \
    --max_sweep_error_deg 4 \
    --axis_mode bounded_tilt \
    --fixed_camera \
    --random \
    --wo_tiny \
    --wo_flow \
    --test_iterations 5000 10000 15000 20000 25000 30000 \
    --save_iterations 5000 10000 15000 20000 25000 30000 \
    --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
    --wandb_mode online \
    --wandb_run_name "leermond-s03-densify13k"

run_job \
    "pepsi-down-s03-densify13k" \
    "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Pepsi_down_und_PNGaR_1846_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    uv run python train.py \
    -s "/shared/datasets/3D_Scanning_MT/cleaned/Session_03/Pepsi_down_und_rn_roi_sqr_PNGaR_1846" \
    --name "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Pepsi_down_und_PNGaR_1846_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    --iterations 30000 \
    --densify_until_iter 13000 \
    --rotation_direction 1 \
    --max_sweep_error_deg 4 \
    --axis_mode bounded_tilt \
    --fixed_camera \
    --random \
    --wo_tiny \
    --wo_flow \
    --test_iterations 5000 10000 15000 20000 25000 30000 \
    --save_iterations 5000 10000 15000 20000 25000 30000 \
    --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
    --wandb_mode online \
    --wandb_run_name "pepsi-down-s03-densify13k"

run_job \
    "pepsi-s03-densify13k" \
    "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Pepsi_und_PNGaR_1846_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    uv run python train.py \
    -s "/shared/datasets/3D_Scanning_MT/cleaned/Session_03/Pepsi_und_rn_roi_sqr_PNGaR_1846" \
    --name "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Pepsi_und_PNGaR_1846_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    --iterations 30000 \
    --densify_until_iter 13000 \
    --rotation_direction 1 \
    --max_sweep_error_deg 4 \
    --axis_mode bounded_tilt \
    --fixed_camera \
    --random \
    --wo_tiny \
    --wo_flow \
    --test_iterations 5000 10000 15000 20000 25000 30000 \
    --save_iterations 5000 10000 15000 20000 25000 30000 \
    --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
    --wandb_mode online \
    --wandb_run_name "pepsi-s03-densify13k"

run_job \
    "quoellfrisch-top-30d-s03-densify13k" \
    "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Quoellfrisch_top_30d_und_PNGaR_1777_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    uv run python train.py \
    -s "/shared/datasets/3D_Scanning_MT/cleaned/Session_03/Quoellfrisch_top_30d_und_rn_roi_sqr_PNGaR_1777" \
    --name "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Quoellfrisch_top_30d_und_PNGaR_1777_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    --iterations 30000 \
    --densify_until_iter 13000 \
    --rotation_direction 1 \
    --max_sweep_error_deg 4 \
    --axis_mode bounded_tilt \
    --fixed_camera \
    --random \
    --wo_tiny \
    --wo_flow \
    --test_iterations 5000 10000 15000 20000 25000 30000 \
    --save_iterations 5000 10000 15000 20000 25000 30000 \
    --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
    --wandb_mode online \
    --wandb_run_name "quoellfrisch-top-30d-s03-densify13k"

run_job \
    "ramseier-tetra-s03-densify13k" \
    "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Ramseier_Tetra_und_PNGaR_3073_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    uv run python train.py \
    -s "/shared/datasets/3D_Scanning_MT/cleaned/Session_03/Ramseier_Tetra_und_rn_roi_sqr_PNGaR_3073" \
    --name "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Ramseier_Tetra_und_PNGaR_3073_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    --iterations 30000 \
    --densify_until_iter 13000 \
    --rotation_direction 1 \
    --max_sweep_error_deg 4 \
    --axis_mode bounded_tilt \
    --fixed_camera \
    --random \
    --wo_tiny \
    --wo_flow \
    --test_iterations 5000 10000 15000 20000 25000 30000 \
    --save_iterations 5000 10000 15000 20000 25000 30000 \
    --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
    --wandb_mode online \
    --wandb_run_name "ramseier-tetra-s03-densify13k_B"

run_job \
    "schwamm-down-s03-densify13k" \
    "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Schwamm_down_und_PNGaR_2114_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    uv run python train.py \
    -s "/shared/datasets/3D_Scanning_MT/cleaned/Session_03/Schwamm_down_und_rn_roi_sqr_PNGaR_2114" \
    --name "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Schwamm_down_und_PNGaR_2114_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    --iterations 30000 \
    --densify_until_iter 13000 \
    --rotation_direction 1 \
    --max_sweep_error_deg 4 \
    --axis_mode bounded_tilt \
    --fixed_camera \
    --random \
    --wo_tiny \
    --wo_flow \
    --test_iterations 5000 10000 15000 20000 25000 30000 \
    --save_iterations 5000 10000 15000 20000 25000 30000 \
    --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
    --wandb_mode online \
    --wandb_run_name "schwamm-down-s03-densify13k"

run_job \
    "schwamm-top-s03-densify13k" \
    "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Schwamm_top_und_PNGaR_1677_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    uv run python train.py \
    -s "/shared/datasets/3D_Scanning_MT/cleaned/Session_03/Schwamm_top_und_rn_roi_sqr_PNGaR_1677" \
    --name "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Schwamm_top_und_PNGaR_1677_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    --iterations 30000 \
    --densify_until_iter 13000 \
    --rotation_direction 1 \
    --max_sweep_error_deg 4 \
    --axis_mode bounded_tilt \
    --fixed_camera \
    --random \
    --wo_tiny \
    --wo_flow \
    --test_iterations 5000 10000 15000 20000 25000 30000 \
    --save_iterations 5000 10000 15000 20000 25000 30000 \
    --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
    --wandb_mode online \
    --wandb_run_name "schwamm-top-s03-densify13k"

run_job \
    "weisswein-s03-densify13k" \
    "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Weisswein_und_PNGaR_1766_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    uv run python train.py \
    -s "/shared/datasets/3D_Scanning_MT/cleaned/Session_03/Weisswein_und_rn_roi_sqr_PNGaR_1766" \
    --name "/shared/datasets/3D_Scanning_MT/output_RotGS/session_03/Weisswein_und_PNGaR_1766_center_xyz_noflow_ccw_densify13k_30k_wandb" \
    --iterations 30000 \
    --densify_until_iter 13000 \
    --rotation_direction 1 \
    --max_sweep_error_deg 4 \
    --axis_mode bounded_tilt \
    --fixed_camera \
    --random \
    --wo_tiny \
    --wo_flow \
    --test_iterations 5000 10000 15000 20000 25000 30000 \
    --save_iterations 5000 10000 15000 20000 25000 30000 \
    --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
    --wandb_mode online \
    --wandb_run_name "weisswein-s03-densify13k"

}

run_queue_pass() {
    QUEUE_ATTEMPT="$1"
    QUEUE_FAILURES=0
    echo "[$(timestamp)] Beginning queue pass $((QUEUE_ATTEMPT + 1)) of $((RETRY_COUNT + 1))."
    run_all_jobs
    echo "[$(timestamp)] Queue pass $((QUEUE_ATTEMPT + 1)) finished with ${QUEUE_FAILURES} incomplete job(s)."
}

run_queue_pass 0

for ((retry = 1; retry <= RETRY_COUNT && QUEUE_FAILURES > 0; retry++)); do
    echo "[$(timestamp)] Waiting ${RETRY_DELAY_SECONDS} seconds before retry pass ${retry}."
    sleep "${RETRY_DELAY_SECONDS}"
    run_queue_pass "${retry}"
done

if (( QUEUE_FAILURES > 0 )); then
    echo "[$(timestamp)] Queue finished with ${QUEUE_FAILURES} job(s) still incomplete after ${RETRY_COUNT} retries."
    exit 1
fi

echo "[$(timestamp)] Queue finished successfully."
