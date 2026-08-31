#!/usr/bin/env bash
set -euo pipefail

readonly GUI_DISPLAY="${DISPLAY:-:1}"
readonly VNC_PORT="${VNC_PORT:-5901}"
readonly NOVNC_PORT="${NOVNC_PORT:-6080}"
readonly GUI_STATE_DIR="${XDG_STATE_HOME:-${HOME}/.local/state}/rotgs-gui"
readonly GUI_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/rotgs-gui-${UID}}"

mkdir -p "${GUI_STATE_DIR}" "${GUI_RUNTIME_DIR}"
chmod 0700 "${GUI_RUNTIME_DIR}"
export DISPLAY="${GUI_DISPLAY}"
export XDG_RUNTIME_DIR="${GUI_RUNTIME_DIR}"

# Serialize concurrent/repeated postStartCommand runs. Services close fd 9 so
# they do not retain the lock after this script exits.
exec 9>"${GUI_RUNTIME_DIR}/start.lock"
flock -w 30 9 || {
    echo "Timed out waiting for another GUI startup run to finish." >&2
    exit 1
}

pid_matches_service() {
    local pid="$1"
    local service="$2"
    local cmdline

    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"

    case "${service}" in
        xvfb)
            [[ "${cmdline}" == *"Xvfb ${GUI_DISPLAY} "* ]]
            ;;
        xfce)
            [[ "${cmdline}" == *"startxfce4"* || "${cmdline}" == *"xfce4-session"* ]]
            ;;
        x11vnc)
            [[ "${cmdline}" == *"x11vnc"* &&
               "${cmdline}" == *"-display ${GUI_DISPLAY} "* &&
               "${cmdline}" == *"-rfbport ${VNC_PORT} "* ]]
            ;;
        websockify)
            [[ "${cmdline}" == *"websockify"* &&
               "${cmdline}" == *"127.0.0.1:${NOVNC_PORT} "* &&
               "${cmdline}" == *"127.0.0.1:${VNC_PORT}"* ]]
            ;;
        *)
            return 1
            ;;
    esac
}

tracked_process_is_running() {
    local pid_file="$1"
    local service="$2"
    local pid=""

    if [[ -r "${pid_file}" ]]; then
        read -r pid < "${pid_file}" || true
    fi

    [[ "${pid}" =~ ^[0-9]+$ ]] &&
        kill -0 "${pid}" 2>/dev/null &&
        pid_matches_service "${pid}" "${service}"
}

is_running() {
    local pid_file="$1"
    local service="$2"

    if tracked_process_is_running "${pid_file}" "${service}"; then
        return 0
    fi

    if [[ -e "${pid_file}" ]]; then
        echo "Removing stale ${service} PID file (${pid_file})." >&2
        rm -f -- "${pid_file}"
    fi
    return 1
}

port_is_open() {
    local port="$1"
    (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null
}

start_detached() {
    local pid_file="$1"
    local log_file="$2"
    shift 2

    # x11vnc installs its own SIGHUP handler, so nohup alone is insufficient.
    # A new session keeps all services outside postStartCommand's process group.
    nohup setsid "$@" </dev/null > "${log_file}" 2>&1 9>&- &
    printf '%s\n' "$!" > "${pid_file}"
}

wait_for_display() {
    local pid_file="$1"

    for ((attempt = 0; attempt < 100; attempt++)); do
        if tracked_process_is_running "${pid_file}" xvfb &&
           xdpyinfo -display "${GUI_DISPLAY}" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.1
    done

    is_running "${pid_file}" xvfb || true
    echo "Xvfb did not make display ${GUI_DISPLAY} available; see ${GUI_STATE_DIR}/xvfb.log" >&2
    return 1
}

wait_for_process() {
    local pid_file="$1"
    local service="$2"

    for ((attempt = 0; attempt < 50; attempt++)); do
        tracked_process_is_running "${pid_file}" "${service}" && return 0
        sleep 0.1
    done

    is_running "${pid_file}" "${service}" || true
    echo "${service} exited during startup; see ${GUI_STATE_DIR}/${service}.log" >&2
    return 1
}

wait_for_port() {
    local pid_file="$1"
    local port="$2"
    local service="$3"

    for ((attempt = 0; attempt < 100; attempt++)); do
        if tracked_process_is_running "${pid_file}" "${service}" && port_is_open "${port}"; then
            return 0
        fi
        sleep 0.1
    done

    is_running "${pid_file}" "${service}" || true
    echo "${service} did not listen on localhost:${port}; see ${GUI_STATE_DIR}/${service}.log" >&2
    return 1
}

start_xvfb() {
    local pid_file="${GUI_RUNTIME_DIR}/xvfb.pid"
    if ! is_running "${pid_file}" xvfb; then
        start_detached "${pid_file}" "${GUI_STATE_DIR}/xvfb.log" \
            Xvfb "${GUI_DISPLAY}" \
            -screen 0 "${XVFB_SCREEN:-1920x1080x24}" \
            -ac -nolisten tcp
    fi

    wait_for_display "${pid_file}"
}

start_xfce() {
    local pid_file="${GUI_RUNTIME_DIR}/xfce.pid"
    if ! is_running "${pid_file}" xfce; then
        start_detached "${pid_file}" "${GUI_STATE_DIR}/xfce.log" \
            dbus-launch --exit-with-session startxfce4
    fi

    wait_for_process "${pid_file}" xfce
}

start_x11vnc() {
    local pid_file="${GUI_RUNTIME_DIR}/x11vnc.pid"
    if ! is_running "${pid_file}" x11vnc; then
        if port_is_open "${VNC_PORT}"; then
            echo "localhost:${VNC_PORT} is already in use by an untracked process; refusing to start x11vnc." >&2
            return 1
        fi
        start_detached "${pid_file}" "${GUI_STATE_DIR}/x11vnc.log" \
            x11vnc \
            -display "${GUI_DISPLAY}" \
            -localhost \
            -forever \
            -shared \
            -nopw \
            -rfbport "${VNC_PORT}" \
            -noxdamage
    fi

    wait_for_port "${pid_file}" "${VNC_PORT}" x11vnc
}

start_novnc() {
    local pid_file="${GUI_RUNTIME_DIR}/websockify.pid"
    if ! is_running "${pid_file}" websockify; then
        if port_is_open "${NOVNC_PORT}"; then
            echo "localhost:${NOVNC_PORT} is already in use by an untracked process; refusing to start websockify." >&2
            return 1
        fi
        start_detached "${pid_file}" "${GUI_STATE_DIR}/websockify.log" \
            websockify \
            --web /usr/share/novnc \
            "127.0.0.1:${NOVNC_PORT}" \
            "127.0.0.1:${VNC_PORT}"
    fi

    wait_for_port "${pid_file}" "${NOVNC_PORT}" websockify
}

start_xvfb
start_xfce
start_x11vnc
start_novnc

echo "RotGS GUI is running on ${GUI_DISPLAY}."
echo "Open the VS Code-forwarded URL: http://localhost:${NOVNC_PORT}/vnc.html?autoconnect=1&resize=remote"
echo "VNC and noVNC listen on container localhost only; use VS Code port forwarding."
