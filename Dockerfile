# Use an official Python runtime as a parent image
FROM davecoulter/astromini3:1.1 as dave_astromini3__1_2
LABEL description="Image for teglon"
LABEL maintainer="Dave Coulter (dcoulter@ucsc.edu)"

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
pip install --upgrade pip && \
conda update -n base conda && \
conda install -c conda-forge shapely && \
apt-get clean && rm -rf /opt/conda/pkgs/* && \
rm -rf /var/lib/apt/lists/* 

CMD source activate root

WORKDIR /app