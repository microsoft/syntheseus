set -x

# Set up LocalRetro first, which RetroKNN depends on.
source setup_shared.sh

# Install extra dependencies specific to RetroKNN.
conda install faiss-gpu -c pytorch -y
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
