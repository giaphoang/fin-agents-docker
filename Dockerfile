FROM ubuntu:24.10
LABEL author="Zap"

# Use SHELL to change default shell to bash for COnda
# login for various conda commands by sourcing both ~/.profile and ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Create a non-root user
# Build arguments
ARG username=zapfinrobot
ARG uid=1204
ARG gid=124
ARG IMAGE_NAME=finrobot
ARG IMAGE_TAG=latest

# Set environment variables (using key=value syntax and braces)
ENV USER=${username} \
    UID=${uid} \
    GID=${gid} \
    HOME=/home/${username} \
    IMAGE_NAME=${IMAGE_NAME} \
    IMAGE_TAG=${IMAGE_TAG}

# Install wget and build tools (including gcc, g++, and python3-dev)
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    python3-dev \
    adduser \
    && ln -s /usr/bin/md5sum /usr/bin/md5

RUN adduser --disabled-password\
    --gecos "Zap Fin Robot user"\
    --uid $UID\
    --gid $GID\
    --home $HOME\
    $USER

#Copy config files
COPY my_conda.yml /tmp/
COPY requirements.txt /tmp/
RUN chown $USER:$GID /tmp/my_conda.yml /tmp/requirements.txt

# libgomp.so.1 issue in aarch64
# ENV LD_PRELOAD="/home/zap/miniconda3/lib/python3.12/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0  /home/zap/miniconda3/lib/python3.12/site-packages/hnswlib.cpython-312-aarch64-linux-gnu.so"

# Switch to non-root user
USER $USER

#ARG for install Miniconda
ARG URL_PREFIX=https://repo.anaconda.com/miniconda
ARG INSTALLER_URL=$URL_PREFIX/Miniconda3-latest-MacOSX-x86_64.sh
ARG CONDA_DIR=$HOME/miniconda3

# Before building the conda environment, remove build strings from my_conda.yml.
# This regex removes any "=<something>_<something>" pattern from each dependency.
RUN sed -E -i 's/(=[^=]+_[^ ]+)//g' /tmp/my_conda.yml && \
    sed -i '/^ *- *libcxx(=.*)?/d' /tmp/my_conda.yml && \
    sed -E -i '/^[[:space:]]*- *appnope(=.*)?/d' /tmp/my_conda.yml

# Remove macOS-specific dependencies (like libcxx) from the environment file
RUN sed -E -i '/^[[:space:]]*- *libcxx(=.*)?/d' /tmp/my_conda.yml

#Install and build Miniconda
ENV URL_PREFIX=${URL_PREFIX} \
    INSTALLER_URL=${INSTALLER_URL} \
    CONDA_DIR=${CONDA_DIR}

RUN wget --quiet $INSTALLER_URL -O ~/miniconda.sh && \
    chmod u+x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH

# make conda activate command available from login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

# make conda-activate command available from interactive shells
RUN conda init bash

# Create a project directory inside user home
ENV PROJECT_DIR=$HOME/finrobot_app
RUN mkdir -p $PROJECT_DIR
WORKDIR $PROJECT_DIR


# build the conda environment
ENV ENV_PREFIX $PROJECT_DIR/env
RUN conda update -n base -c defaults conda && \
    conda env create -p $ENV_PREFIX -f /tmp/my_conda.yml python=3.10.16 && \
    conda clean --all --yes

# install any JupyterLab extensions (optional!)
RUN conda activate $ENV_PREFIX 

# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Install pip requirements 
RUN pip install -r /tmp/requirements.txt

# # Create directories for vector databases
# RUN mkdir -p $PROJECT_DIR/earnings-call-db $PROJECT_DIR/sec-filings-db $PROJECT_DIR/sec-filings-md-db $PROJECT_DIR/report

# install some packages in case cannot import (rag_up)
# RUN pip install sentence-transformers -q
# RUN pip install langchain-chroma -U -q

# Copy your ASGI application file into the project directory.
# Make sure main_up.py exists in your build context.
COPY main_up.py $PROJECT_DIR/
COPY finrobot $PROJECT_DIR/finrobot
COPY report $PROJECT_DIR/report
COPY config_api_keys $PROJECT_DIR/config_api_keys
COPY OAI_CONFIG_LIST $PROJECT_DIR/OAI_CONFIG_LIST


# Expose the API port
EXPOSE 80

# Environment variables to avoid TensorFlow warnings and configure behavior
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV DISABLE_TENSORFLOW_LOGS=1

# Activate conda environment and run the application
SHELL ["/bin/bash", "--login", "-c"]
CMD ["uvicorn", "main_up:app", "--host", "0.0.0.0", "--port", "80"]

 