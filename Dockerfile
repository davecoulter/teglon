# Use an official Python runtime as a parent image
FROM davecoulter/astromini3:1.4 as dave_astromini3__1_5
LABEL description="Image for teglon debug"
LABEL maintainer="Dave Coulter (dcoulter@ucsc.edu)"

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
apt-get install htop && \
pip install --upgrade pip && \
pip install mysqlclient && \
#pip install ray==0.7.6 && \
conda install -c conda-forge spherical-geometry && \
conda update numpy && \
apt-get clean && rm -rf /opt/conda/pkgs/* && \
rm -rf /var/lib/apt/lists/*

WORKDIR /app