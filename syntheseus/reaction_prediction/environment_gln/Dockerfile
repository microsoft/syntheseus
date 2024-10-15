FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04
MAINTAINER krmaziar@microsoft.com

# Set bash, as conda doesn't like dash
SHELL [ "/bin/bash", "--login", "-c" ]

# Make bash aware of conda
RUN echo ". /opt/miniconda/etc/profile.d/conda.sh" >> ~/.profile

# Turn off caching in pip
ENV PIP_NO_CACHE_DIR=1

# Install the dependencies into conda's default environment
COPY ./environment.yml /tmp/
RUN conda config --remove channels defaults && \
    conda config --add channels conda-forge && \
    conda update --all && \
    conda install mamba -n base -c conda-forge && \
    conda clean --all --yes
RUN mamba env update -p /opt/miniconda -f /tmp/environment.yml && conda clean --all --yes

# Install RDKit from source
RUN git clone https://github.com/rdkit/rdkit.git
WORKDIR /rdkit
RUN git checkout 8bd74d91118f3fdb370081ef0a18d71715e7c6cf
RUN mkdir build && cd build && cmake -DPy_ENABLE_SHARED=1 \
    -DRDK_INSTALL_INTREE=ON \
    -DRDK_INSTALL_STATIC_LIBS=OFF \
    -DRDK_BUILD_CPP_TESTS=ON \
    -DPYTHON_NUMPY_INCLUDE_PATH="$(python -c 'import numpy ; print(numpy.get_include())')" \
    -DBOOST_ROOT="$CONDA_PREFIX" \
    .. && make && make install
WORKDIR /

# Install GLN (this relies on `CUDA_HOME` being set correctly)
RUN git clone https://github.com/kmaziarz/GLN.git
WORKDIR /GLN
RUN pip install -e .

ENV PYTHONPATH=$PYTHONPATH:/rdkit:/GLN
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/rdkit/lib
