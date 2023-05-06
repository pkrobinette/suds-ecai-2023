#
# Sanitizing Universal and Dependent Steganography
#

# Define base image/operating system
FROM python:3.8

# ENV DEBIAN_FRONTEND=noninteractive

# Install software
RUN apt-get update -y \
    && apt-get install -y python3-pip

# Set container's working directory
WORKDIR /suds-ecai-2023

# Copy files and directory structure to working directory
COPY . .

# Install necessary packages for SUDS
RUN pip install -r requirements.txt

# Run commands specified in "run.sh" to get started
# ENTRYPOINT ["sh", "scripts/test_all.sh"]
