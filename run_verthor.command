#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
PYTHON_BIN="${PYTHON_BIN:-python3}"

choose_python_bin() {
  local candidate

  if [[ -n "${PYTHON_BIN:-}" ]] && command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "$PYTHON_BIN"
    return
  fi

  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return
    fi
  done

  return 1
}

show_error() {
  local message="$1"
  osascript -e "display alert \"Verthor Error\" message \"${message}\" as critical"
}

show_info() {
  local message="$1"
  osascript -e "display notification \"${message}\" with title \"Verthor\""
}

confirm_run_settings() {
  local input_path="$1"
  local output_path="$2"
  local preset="$3"
  local saliency_model="$4"

  osascript <<APPLESCRIPT
button returned of (display dialog "Input: ${input_path}
Output: ${output_path}
Preset: ${preset}
Saliency: ${saliency_model}

Run with these settings?" buttons {"Cancel", "Run"} default button "Run")
APPLESCRIPT
}

choose_input_file() {
  osascript <<'APPLESCRIPT'
POSIX path of (choose file with prompt "Select a video for Verthor")
APPLESCRIPT
}

choose_preset() {
  osascript <<'APPLESCRIPT'
set presetChoice to choose from list {"talking_head", "sports", "pets", "cars", "movie"} with prompt "Choose a framing preset" default items {"talking_head"}
if presetChoice is false then
  return ""
end if
return item 1 of presetChoice
APPLESCRIPT
}

choose_debug_preview() {
  osascript <<'APPLESCRIPT'
button returned of (display dialog "Create debug preview video too?" buttons {"No", "Yes"} default button "No")
APPLESCRIPT
}

choose_saliency_model() {
  osascript <<'APPLESCRIPT'
set saliencyChoice to choose from list {"handcrafted", "deepgazemr"} with prompt "Choose saliency mode (handcrafted is faster; deepgazemr is slower and experimental)" default items {"handcrafted"}
if saliencyChoice is false then
  return ""
end if
return item 1 of saliencyChoice
APPLESCRIPT
}

build_output_path() {
  local input_path="$1"
  local input_dir="${input_path:h}"
  local input_name="${input_path:t}"
  local stem="${input_name:r}"
  echo "$input_dir/${stem}_vertical.mp4"
}

normalize_input_path() {
  local input_path="$1"
  local abs_input_path=""
  local script_path="$SCRIPT_DIR/${0:t}"

  if [[ -z "$input_path" ]]; then
    echo ""
    return
  fi

  if [[ -e "$input_path" ]]; then
    abs_input_path="$(cd "${input_path:h}" && pwd)/${input_path:t}"
  fi

  if [[ "$input_path" == "$0" || "$input_path" == "$script_path" ]]; then
    echo ""
    return
  fi

  if [[ -n "$abs_input_path" && "$abs_input_path" == "$script_path" ]]; then
    echo ""
    return
  fi

  if [[ "${input_path:e:l}" == "command" ]]; then
    echo ""
    return
  fi

  echo "$input_path"
}

is_probably_video_file() {
  local input_path="$1"
  local ext="${input_path:e:l}"

  case "$ext" in
    mp4|mov|m4v|avi|mkv|webm|mpg|mpeg|mp2|m2v|mts|m2ts|ts|wmv|flv|ogv)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

ensure_ffmpeg() {
  if ! command -v ffmpeg >/dev/null 2>&1; then
    show_error "ffmpeg is not installed or not in PATH. Install it with: brew install ffmpeg"
    exit 1
  fi
}

ensure_python() {
  PYTHON_BIN="$(choose_python_bin || true)"
  if [[ -z "$PYTHON_BIN" ]]; then
    show_error "python3 was not found in PATH. Install Python 3.11+ first."
    exit 1
  fi
}

setup_environment() {
  cd "$SCRIPT_DIR"

  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi

  if [[ ! -x "$VENV_PYTHON" ]]; then
    show_error "Virtual environment is incomplete. Delete .venv and run again."
    exit 1
  fi

  mkdir -p "$VENV_DIR/cache/matplotlib" "$VENV_DIR/cache/xdg"
  export MPLCONFIGDIR="$VENV_DIR/cache/matplotlib"
  export XDG_CACHE_HOME="$VENV_DIR/cache/xdg"

  if ! "$VENV_PYTHON" -c "import verthor" >/dev/null 2>&1; then
    "$VENV_PYTHON" -m pip install --no-build-isolation -e .
  fi
}

main() {
  ensure_python
  ensure_ffmpeg

  local input_path="${1:-}"
  input_path="$(normalize_input_path "$input_path")"
  if [[ -z "$input_path" ]]; then
    input_path="$(choose_input_file)"
  fi

  if [[ -z "$input_path" ]]; then
    exit 0
  fi

  if [[ ! -f "$input_path" ]]; then
    show_error "Selected file does not exist: $input_path"
    exit 1
  fi

  if ! is_probably_video_file "$input_path"; then
    input_path="$(choose_input_file)"
  fi

  if [[ -z "$input_path" ]]; then
    exit 0
  fi

  if [[ ! -f "$input_path" ]]; then
    show_error "Selected file does not exist: $input_path"
    exit 1
  fi

  local preset
  preset="$(choose_preset)"
  if [[ -z "$preset" ]]; then
    exit 0
  fi

  local debug_choice
  debug_choice="$(choose_debug_preview)"

  local saliency_model
  saliency_model="$(choose_saliency_model)"
  if [[ -z "$saliency_model" ]]; then
    exit 0
  fi

  local output_path
  output_path="$(build_output_path "$input_path")"

  local -a cmd
  cmd=(
    "$VENV_PYTHON" -m verthor "$input_path" "$output_path"
    --preset "$preset"
    --saliency-model "$saliency_model"
  )

  if [[ "$debug_choice" == "Yes" ]]; then
    cmd+=(--save-debug-preview)
  fi

  echo "Project directory: $SCRIPT_DIR"
  echo "Input: $input_path"
  echo "Output: $output_path"
  echo "Preset: $preset"
  echo "Saliency: $saliency_model"
  echo ""

  local confirmation
  confirmation="$(confirm_run_settings "$input_path" "$output_path" "$preset" "$saliency_model")"
  if [[ "$confirmation" != "Run" ]]; then
    exit 0
  fi

  setup_environment

  printf 'Running command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  "${cmd[@]}"

  show_info "Processing finished: ${output_path:t}"
  open -R "$output_path"
}

main "$@"
