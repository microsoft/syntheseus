set -x

# Install extra dependencies specific to RootAligned.
pip install OpenNMT-py==2.2.0 textdistance==4.2.2

export GITHUB_ORG_NAME=otori-bird
export GITHUB_REPO_NAME=retrosynthesis
export GITHUB_COMMIT_ID=ea3b5729752fdc319b18ea4c65c1a573e24d7320

source setup_shared.sh
