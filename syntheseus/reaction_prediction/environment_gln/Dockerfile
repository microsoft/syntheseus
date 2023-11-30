FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.0-cudnn7-ubuntu18.04
MAINTAINER krmaziar@microsoft.com

# Set bash, as conda doesn't like dash
SHELL [ "/bin/bash", "--login", "-c" ]

# Make bash aware of conda
RUN echo ". /opt/miniconda/etc/profile.d/conda.sh" >> ~/.profile

# Turn off caching in pip
ENV PIP_NO_CACHE_DIR=1

# Install the dependencies into conda's default environment
COPY ./environment.yml /tmp/
RUN conda install mamba -n base -c conda-forge
RUN mamba env update -p /opt/miniconda -f /tmp/environment.yml && conda clean -ay

# Install RDKit from source
RUN git clone https://github.com/rdkit/rdkit.git
WORKDIR /rdkit
RUN git checkout 7ad9e0d161110f758350ca080be0fc05530bee1e
RUN mkdir build && cd build && cmake -DPy_ENABLE_SHARED=1 \
    -DRDK_INSTALL_INTREE=ON \
    -DRDK_INSTALL_STATIC_LIBS=OFF \
    -DRDK_BUILD_CPP_TESTS=ON \
    -DPYTHON_NUMPY_INCLUDE_PATH="$(python -c 'import numpy ; print(numpy.get_include())')" \
    -DBOOST_ROOT="$CONDA_PREFIX" \
    .. && make && make install
WORKDIR /

# Install GLN (this relies on `CUDA_HOME` being set correctly).
RUN git clone https://github.com/Hanjun-Dai/GLN.git
WORKDIR /GLN
RUN git checkout b5bd7b181a61a8289cc1d1a33825b2c417bed0ef
RUN pip install -e .

ENV PYTHONPATH=$PYTHONPATH:/rdkit:/GLN
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/rdkit/lib
