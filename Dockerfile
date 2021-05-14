FROM davecoulter/astromini3:1.5
LABEL description="Image for teglon debug"
LABEL maintainer="Dave Coulter (dcoulter@ucsc.edu)"

SHELL ["/bin/bash", "-c"]

# If there are problems, remove astLib and htop... they aren't necessary right now
RUN apt-get update && \
pip install --upgrade pip && \
apt-get clean && rm -rf /opt/conda/pkgs/* && \
rm -rf /var/lib/apt/lists/*

WORKDIR /app