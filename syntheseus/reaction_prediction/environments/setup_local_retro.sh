#!/bin/bash

# Install extra dependencies specific to LocalRetro.
conda install dgl-cuda11.3 -c dglteam -y
pip install dgllife chardet

export GITHUB_ORG_NAME=kaist-amsg
export GITHUB_REPO_NAME=LocalRetro
export GITHUB_REPO_DIR=local_retro
export GITHUB_COMMIT_ID=28aa215236c20e719fa4c977089c62fef551adf2

source setup_shared.sh
