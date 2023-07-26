set -x

# Install extra dependencies specific to Chemformer.
pip install pytorch-lightning==1.9.4 git+https://github.com/MolecularAI/pysmilesutils.git

export GITHUB_ORG_NAME=MolecularAI
export GITHUB_REPO_NAME=Chemformer
export GITHUB_COMMIT_ID=6333badcd4e1d92891d167426c96c70f5712ecc3

source setup_shared.sh
