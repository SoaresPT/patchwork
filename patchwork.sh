#!/usr/bin/env bash

# Grab current directory of this script
DIR="$( cd "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" >/dev/null 2>&1 && pwd )"

# Detect the operating system
OS_TYPE="$(uname -s)"

# Enable virtual environment based on OS
case "$OS_TYPE" in
    Linux*|Darwin*)
        # For Linux and macOS
        source "$DIR/.venv/bin/activate"
        ;;
    CYGWIN*|MINGW32*|MSYS*|MINGW*)
        # For Windows (Git Bash)
        source "$DIR/.venv/Scripts/activate"
        ;;
    *)
        echo "Unsupported OS: $OS_TYPE"
        exit 1
        ;;
esac

# Run main.py with the provided directory path
python "$DIR/main.py" "$@"