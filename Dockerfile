FROM ubuntu:20.04
LABEL author="Zap"

# Use SHELL to change default shell to bash for COnda
# login for various conda commands by sourcing both ~/.profile and ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Create a non-root user
ARG username=zapfinrobot
ARG uid=1204
ARG gid=124
# Set image name and tag for the container
ARG IMAGE_NAME=finrobot
ARG IMAGE_TAG=latest

ENV USER=$username UID=$uid GID=$gid HOME=/home/$USER IMAGE_NAME=$IMAGE_NAME IMAGE_TAG=$IMAGE_TAG

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

# Switch to non-root user
USER $USER

#Install Miniconda
ENV URL_PREFIX=https://repo.anaconda.com/miniconda \
    INSTALLER_URL=$URL_PREFIX/Miniconda3-latest-Linux-x86_64.sh \
    CONDA_DIR=~/miniconda3

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
ENV PROJECT_DIR ~/finrobot_app
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

# build the conda environment
ENV ENV_PREFIX $PROJECT_DIR/env
RUN conda update -n base -c defaults conda && \
    conda env create -p $ENV_PREFIX -f /tmp/my_conda.yml && \
    conda clean --all --yes

# install any JupyterLab extensions (optional!)
RUN conda activate $ENV_PREFIX 

# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Install pip requirements 
RUN pip install -r /tmp/requirements.txt

# # Create directories for vector databases
# RUN mkdir -p $PROJECT_DIR/earnings-call-db $PROJECT_DIR/sec-filings-db $PROJECT_DIR/sec-filings-md-db $PROJECT_DIR/report


# Expose the API port
EXPOSE 8888

# Environment variables to avoid TensorFlow warnings and configure behavior
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV DISABLE_TENSORFLOW_LOGS=1

# Activate conda environment and run the application
SHELL ["/bin/bash", "--login", "-c"]
CMD ["conda", "run", "-p", "$ENV_PREFIX", "uvicorn", "main_up:app", "--host", "0.0.0.0", "--port", "8888"]
 