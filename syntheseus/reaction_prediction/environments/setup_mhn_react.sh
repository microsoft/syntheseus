set -x

# Install extra dependencies specific to MHNreact.
conda install rdchiral_cpp -c conda-forge -y
pip install scikit-learn scipy swifter tqdm wandb

# Install our fork of the open-source MHNreact code, which includes some efficiency improvements.
pip install git+https://github.com/kmaziarz/mhn-react.git
