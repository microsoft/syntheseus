#!/bin/bash

# Install extra dependencies specific to LocalRetro.
conda install dgl-cuda11.3 -c dglteam -y
pip install dgllife chardet

export GITHUB_ORG_NAME=kaist-amsg
export GITHUB_REPO_NAME=LocalRetro
export GITHUB_COMMIT_ID=7dab59f7f85eca8b1c04c18fe8575fb1568ff7ae

source setup_shared.sh
