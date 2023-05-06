#
# Sanitizing Universal and Dependent Steganography
#

# Define base image/operating system
FROM python:3.8

# ENV DEBIAN_FRONTEND=noninteractive

# Install software
RUN apt-get update -y \
    && apt-get install -y python3-pip

# Accept a build argument for the working directory
ARG WORKDIR_PATH

# Set the environment variable for the working directory
ENV WORKDIR_PATH ${WORKDIR_PATH}

# Set container's working directory
WORKDIR ${WORKDIR_PATH}

# Copy files and directory structure to working directory
COPY . .

# Install necessary packages for SUDS
RUN pip install -r requirements.txt

# Run commands specified in "run.sh" to get started
# ENTRYPOINT ["sh", "scripts/test_all.sh"]
