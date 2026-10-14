#!/usr/bin/env bash

# Sequential, disconnect-safe RotGS queue for the remaining Session_04
# Quoellfrisch Top SAM31AC ablations. Launch with nohup. Before each run, GPU 0
# must pass five consecutive idle checks. Only one training job runs at a time.

set -uo pipefail
# Keep background children in the queue's group until setsid creates their
# session. This prevents setsid from forking, so $! remains the session leader.
set +m

readonly REPO_ROOT="/workspaces/RotGS"
readonly LOG_ROOT="${REPO_ROOT}/logs/train_queue/session_04_quoellfrisch_top_sam31ac_ablation"
readonly LOCK_FILE="${REPO_ROOT}/.train_queue_gpu0.lock"
readonly GPU_INDEX=0
readonly FINAL_ITERATION=30000
# Observed maximum full-orbit error for the nominal-angle runs.
readonly GLOBAL_SWEEP_BOUND_DEG=1.2
readonly RETRY_COUNT=2
readonly RETRY_DELAY_SECONDS=300
readonly MIN_FREE_KIB=$((10 * 1024 * 1024))
readonly IGNORED_GPU_PROCESS_MAX_MIB=0
readonly GPU_IDLE_MAX_UTILIZATION=5
readonly GPU_IDLE_REQUIRED_CHECKS=5
readonly GPU_POLL_SECONDS=2
readonly GPU_MONITOR_LOG="${LOG_ROOT}/gpu-monitor.log"

QUEUE_ATTEMPT=0
QUEUE_FAILURES=0
DELAY_MINUTES=0
DRY_RUN=0

usage() {
    echo "Usage: $0 [--delay-minutes MINUTES] [--dry-run]"
}

while (( $# > 0 )); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
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
readonly DRY_RUN

cd "${REPO_ROOT}" || exit 1
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

if (( DRY_RUN == 0 )); then
    mkdir -p "${LOG_ROOT}"
    if ! command -v flock >/dev/null 2>&1; then
        echo "ERROR: flock is required but is not installed." >&2
        exit 1
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi is required but is not installed." >&2
        exit 1
    fi

    if ! command -v setsid >/dev/null 2>&1; then
        echo "ERROR: setsid is required but is not installed." >&2
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
fi

timestamp() {
    date --iso-8601=seconds
}

gpu_process_snapshot() {
    nvidia-smi \
        --id="${GPU_INDEX}" \
        --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null
}

foreign_gpu_processes() {
    local process_snapshot="$1"
    local own_pgid="${2:-}"
    local pid
    local process_name
    local used_memory
    local process_pgid
    local result=""

    while IFS=',' read -r pid process_name used_memory; do
        pid="${pid//[[:space:]]/}"
        used_memory="${used_memory//[[:space:]]/}"
        [[ -n "${pid}" ]] || continue

        if [[ -n "${own_pgid}" ]]; then
            process_pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null || true)"
            process_pgid="${process_pgid//[[:space:]]/}"
            if [[ "${process_pgid}" == "${own_pgid}" ]]; then
                continue
            fi
        fi

        # Any foreign CUDA allocation blocks startup, even a small one.
        if [[ "${used_memory}" =~ ^[0-9]+$ ]] \
            && (( used_memory <= IGNORED_GPU_PROCESS_MAX_MIB )); then
            continue
        fi

        result+="${pid}:${process_name}:${used_memory}MiB; "
    done <<< "${process_snapshot}"

    printf '%s' "${result}"
}

log_gpu_snapshot() {
    local label="$1"
    local process_snapshot="$2"
    local gpu_stats
    local process_summary

    if ! gpu_stats="$(
        nvidia-smi \
            --id="${GPU_INDEX}" \
            --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
            --format=csv,noheader,nounits 2>/dev/null
    )"; then
        gpu_stats="query-failed"
    fi

    process_summary="${process_snapshot//$'\n'/; }"
    [[ -n "${process_summary}" ]] || process_summary="none"
    printf '[%s] job=%s gpu=%s processes=%s\n' \
        "$(timestamp)" "${label}" "${gpu_stats}" "${process_summary}" \
        >>"${GPU_MONITOR_LOG}"
}

wait_for_gpu() {
    local process_snapshot
    local foreign_processes
    local gpu_utilization
    local idle_checks=0

    while true; do
        if ! process_snapshot="$(gpu_process_snapshot)"; then
            idle_checks=0
            echo "[$(timestamp)] ERROR: could not query GPU state; retrying in ${GPU_POLL_SECONDS} seconds."
            sleep "${GPU_POLL_SECONDS}"
            continue
        fi

        log_gpu_snapshot "waiting" "${process_snapshot}"
        foreign_processes="$(foreign_gpu_processes "${process_snapshot}")"
        if ! gpu_utilization="$(
            nvidia-smi --id="${GPU_INDEX}" \
                --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null
        )"; then
            idle_checks=0
            echo "[$(timestamp)] ERROR: could not query GPU utilization; retrying in ${GPU_POLL_SECONDS} seconds."
            sleep "${GPU_POLL_SECONDS}"
            continue
        fi
        gpu_utilization="${gpu_utilization//[[:space:]]/}"
        if [[ -z "${foreign_processes}" \
            && "${gpu_utilization}" =~ ^[0-9]+$ \
            && "${gpu_utilization}" -le "${GPU_IDLE_MAX_UTILIZATION}" ]]; then
            idle_checks=$((idle_checks + 1))
            echo "[$(timestamp)] GPU ${GPU_INDEX} idle check ${idle_checks}/${GPU_IDLE_REQUIRED_CHECKS}."
            if (( idle_checks >= GPU_IDLE_REQUIRED_CHECKS )); then
                return
            fi
        else
            idle_checks=0
            echo "[$(timestamp)] Waiting for GPU ${GPU_INDEX}: utilization=${gpu_utilization:-unknown}% " \
                "(idle <=${GPU_IDLE_MAX_UTILIZATION}%), foreign processes=${foreign_processes:-none}; " \
                "idle streak reset, retrying in ${GPU_POLL_SECONDS} seconds."
        fi

        sleep "${GPU_POLL_SECONDS}"
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

signal_job_group() {
    local signal_name="$1"
    local job_pid="$2"
    local queue_pgid
    local job_identity
    local actual_pgid
    local actual_sid

    [[ "${job_pid}" =~ ^[1-9][0-9]*$ ]] || return 1
    if ! queue_pgid="$(ps -o pgid= -p "$$" 2>/dev/null)"; then
        return 1
    fi
    queue_pgid="${queue_pgid//[[:space:]]/}"
    [[ "${queue_pgid}" =~ ^[1-9][0-9]*$ \
        && "${job_pid}" != "${queue_pgid}" ]] || return 1
    if ! job_identity="$(ps -o pgid=,sid= -p "${job_pid}" 2>/dev/null)"; then
        return 1
    fi
    read -r actual_pgid actual_sid <<< "${job_identity}"
    # Signal only the isolated session created for this job. Never target
    # the queue's group, including while setsid is still starting up.
    [[ "${actual_pgid}" == "${job_pid}" \
        && "${actual_sid}" == "${job_pid}" ]] || return 1
    kill "-${signal_name}" -- "-${job_pid}"
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
    local job_pid
    local job_pgid
    local job_paused=0
    local process_snapshot
    local foreign_processes
    local -a command=("$@")

    if (( DRY_RUN == 1 )); then
        printf 'Job: %s\nOutput: %s\nCommand:' "${label}" "${output_path}"
        printf ' %q' "${command[@]}"
        printf '\n\n'
        return 0
    fi

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
    printf '\nLog: %s\nGPU monitor: %s\n' "${log_path}" "${GPU_MONITOR_LOG}"

    # Use a separate process group so only this RotGS job is paused/resumed.
    setsid --wait "${command[@]}" >"${log_path}" 2>&1 &
    job_pid=$!
    # setsid makes this PID the new process-group/session ID. Querying ps
    # immediately after launch can still return the queue's group, allowing
    # a later SIGSTOP to suspend the queue itself. Never cache that group.
    job_pgid="${job_pid}"

    while kill -0 "${job_pid}" 2>/dev/null; do
        if ! process_snapshot="$(gpu_process_snapshot)"; then
            printf '[%s] job=%s gpu=query-failed processes=query-failed\n' \
                "$(timestamp)" "${label}" >>"${GPU_MONITOR_LOG}"
            sleep "${GPU_POLL_SECONDS}"
            continue
        fi

        log_gpu_snapshot "${label}" "${process_snapshot}"
        foreign_processes="$(
            foreign_gpu_processes "${process_snapshot}" "${job_pgid}"
        )"

        if [[ -n "${foreign_processes}" ]] && (( job_paused == 0 )); then
            if signal_job_group STOP "${job_pid}"; then
                job_paused=1
                echo "[$(timestamp)] PAUSE ${label}: foreign GPU " \
                    "process(es): ${foreign_processes}"
            fi
        elif [[ -z "${foreign_processes}" ]] && (( job_paused == 1 )); then
            if signal_job_group CONT "${job_pid}"; then
                job_paused=0
                echo "[$(timestamp)] RESUME ${label}: GPU has no foreign CUDA allocations."
            fi
        fi

        sleep "${GPU_POLL_SECONDS}"
    done

    # Ensure a stopped child can be reaped if it exited during a race.
    if (( job_paused == 1 )); then
        signal_job_group CONT "${job_pid}" || true
    fi

    if wait "${job_pid}"; then
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

run_legacy_session03_jobs() {

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

run_all_jobs() {
    local top_dataset="/shared/datasets/3D_Scanning_MT/cleaned/Session_04/4_Quoellfrisch_Top_30d_SAM31AC_1435"
    local output_root="/shared/datasets/3D_Scanning_MT/output_RotGS/session_04"
    # Six-variant experiment plan. The first two runs are already supplied
    # separately and are deliberately excluded from this queue:
    # 1. TAE, LR, global +/-0.15 deg:
    #    4_Quoellfrisch_Top_30d_SAM31AC_1435_TAE_local_residual_30k
    # 2. TAE, noLR, global +/-0.15 deg:
    #    4_Quoellfrisch_Top_30d_SAM31AC_1435_TAE_wo_tiny_30k
    # Queue order: 4. TAE/LR/noGlobal, 3. TAE/noLR/noGlobal,
    #              5. noTAE/LR, 6. noTAE/noLR.
    # --wo_tiny disables local residuals only. A zero sweep bound disables
    # the global correction; a nonzero bound keeps it active even with noLR.
    # TAE also initializes the axis and center from calibration; noTAE uses
    # the training code's nominal geometry defaults.
    local -a common_args=(
        --iterations "${FINAL_ITERATION}"
        --densify_until_iter 13000
        --resolution 1
        --rotation_direction 1
        --angle_noise_std 0
        --max_residual_angle_deg 0.05
        --center_max_offset 0.05
        --max_gaussians 450000
        --fixed_camera
        --random
        --wo_flow
        --test_iterations 5000 10000 15000 20000 25000 30000
        --save_iterations 5000 10000 15000 20000 25000 30000
        --checkpoint_iterations 5000 10000 15000 20000 25000 30000
        --wandb_mode online
    )

    # 4. Measured TAE angles, local residuals +/-0.05 deg, no global correction.
    run_job \
        "quoellfrisch-top-s04-sam31ac1435-tae-local-residual-no-global" \
        "${output_root}/4_Quoellfrisch_Top_30d_SAM31AC_1435_TAE_local_residual_no_global_30k" \
        uv run --no-sync python train.py \
        -s "${top_dataset}" \
        --name "${output_root}/4_Quoellfrisch_Top_30d_SAM31AC_1435_TAE_local_residual_no_global_30k" \
        "${common_args[@]}" \
        --leverage_TAE \
        --max_sweep_error_deg 0 \
        --wandb_run_name "quoellfrisch-top-s04-sam31ac1435-tae-local-residual-no-global"

    # 3. Keep measured TAE angles fixed: no local or global correction.
    run_job \
        "quoellfrisch-top-s04-sam31ac1435-tae-wo-tiny-no-global" \
        "${output_root}/4_Quoellfrisch_Top_30d_SAM31AC_1435_TAE_wo_tiny_no_global_30k" \
        uv run --no-sync python train.py \
        -s "${top_dataset}" \
        --name "${output_root}/4_Quoellfrisch_Top_30d_SAM31AC_1435_TAE_wo_tiny_no_global_30k" \
        "${common_args[@]}" \
        --leverage_TAE --wo_tiny \
        --max_sweep_error_deg 0 \
        --wandb_run_name "quoellfrisch-top-s04-sam31ac1435-tae-wo-tiny-no-global"

    # 5. Nominal angles, local residuals +/-0.05 deg, global +/-1.2 deg.
    run_job \
        "quoellfrisch-top-s04-sam31ac1435-noTAE-LR" \
        "${output_root}/4_Quoellfrisch_Top_30d_SAM31AC_1435_noTAE_LR_30k" \
        uv run --no-sync python train.py \
        -s "${top_dataset}" \
        --name "${output_root}/4_Quoellfrisch_Top_30d_SAM31AC_1435_noTAE_LR_30k" \
        "${common_args[@]}" \
        --max_sweep_error_deg "${GLOBAL_SWEEP_BOUND_DEG}" \
        --wandb_run_name "quoellfrisch-top-s04-sam31ac1435-noTAE-LR"

    # 6. Nominal angles, no local residuals, global +/-1.2 deg.
    run_job \
        "quoellfrisch-top-s04-sam31ac1435-noTAE-noLR" \
        "${output_root}/4_Quoellfrisch_Top_30d_SAM31AC_1435_noTAE_noLR_30k" \
        uv run --no-sync python train.py \
        -s "${top_dataset}" \
        --name "${output_root}/4_Quoellfrisch_Top_30d_SAM31AC_1435_noTAE_noLR_30k" \
        "${common_args[@]}" \
        --wo_tiny \
        --max_sweep_error_deg "${GLOBAL_SWEEP_BOUND_DEG}" \
        --wandb_run_name "quoellfrisch-top-s04-sam31ac1435-noTAE-noLR"
}

run_queue_pass() {
    QUEUE_ATTEMPT="$1"
    QUEUE_FAILURES=0
    echo "[$(timestamp)] Beginning queue pass $((QUEUE_ATTEMPT + 1)) of $((RETRY_COUNT + 1))."
    run_all_jobs
    echo "[$(timestamp)] Queue pass $((QUEUE_ATTEMPT + 1)) finished with ${QUEUE_FAILURES} incomplete job(s)."
}

if (( DRY_RUN == 1 )); then
    echo "Dry run: GPU ${GPU_INDEX}, ${GPU_IDLE_REQUIRED_CHECKS} consecutive idle checks (<=${GPU_IDLE_MAX_UTILIZATION}%, no foreign CUDA allocations); poll every ${GPU_POLL_SECONDS}s."
    run_all_jobs
    exit 0
fi

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
