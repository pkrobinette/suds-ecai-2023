#
# Self-Preserving Genetic Algorithms
#

FROM python:3.9

# set the user to root
USER root

# set working directory to spga
WORKDIR /spga

# install pip
RUN set -xe \
    && apt-get update -y \
    && apt-get install -y python3-pip
RUN pip install --upgrade pip

# set environment variables
ENV PYTHONPATH='/spga'
# ENV OPENBLAS_NUM_THREADS=1
# ENV OMP_NUM_THREADS=1

# copy files to docker
COPY . .

# install python package dependencies
RUN pip install .
# RUN pip3 install seaborn -U
