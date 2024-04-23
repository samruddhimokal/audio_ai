#!/bin/bash

# Create a Python 3.10 virtual environment named "autodiarize"
python3.10 -m venv autodiarize

# Activate the virtual environment
source autodiarize/bin/activate

apt update && upgrade

apt install ffmpeg

apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython
pip install nemo_toolkit['all']
pip install wget
pip install srt
pip install "-e git+https://github.com/m-bain/whisperX@78dcfaab51005aa703ee21375f81ed31bc248560#egg=whisperx"
pip install -r requirements.txt
pip install yt_dlp
