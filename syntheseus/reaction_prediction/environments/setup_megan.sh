#!/bin/bash

# Install extra dependencies specific to MEGAN.
pip install gin-config==0.3.0 tensorflow==2.13.0 torchtext==0.13.1

export GITHUB_ORG_NAME=molecule-one
export GITHUB_REPO_NAME=megan
export GITHUB_REPO_DIR=$GITHUB_REPO_NAME
export GITHUB_COMMIT_ID=bd6179e42052521e46728adb2bb80dea6905bf40

source setup_shared.sh
