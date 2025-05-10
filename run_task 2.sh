#!/bin/bash
# Simple wrapper script with timeout
timeout 300s uv run src/main.py --task "$1" --verbose False --output_path "$2"
exit $? 