#!/bin/bash

# Make a subdirectory for storing downloaded external repositories.
mkdir -p external

# Add the `external/` directory to `PYTHONPATH` when the environment is activated.
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "export PYTHONPATH=$PWD/external:.:$PYTHONPATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

export GITHUB_NAME="$GITHUB_ORG_NAME/$GITHUB_REPO_NAME"
export MODEL_DIR="external/$GITHUB_REPO_DIR"

echo "Setting up $GITHUB_NAME under $MODEL_DIR"
git -C external clone "https://github.com/$GITHUB_NAME.git" $GITHUB_REPO_DIR
git -C $MODEL_DIR checkout $GITHUB_COMMIT_ID
