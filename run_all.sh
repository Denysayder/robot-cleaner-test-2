#!/bin/bash

source .venv/bin/activate

python3 view_video.py &
python3 realtime_comparison.py &

wait