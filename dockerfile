# This is a version of the dockerfile from the pixi documentation
#
# Modified by Sharon Batiste on 2/26/2024
# Modified for use with CoastSeg

# FROM ghcr.io/prefix-dev/pixi:0.41.4 AS build
FROM ghcr.io/prefix-dev/pixi:latest

# copy source code, pixi.toml and pixi.lock to the container
# make coastseg a working directory
WORKDIR /coastseg

# COPY the license and readme files otherwise the build will fail
COPY ./LICENSE ./LICENSE 
COPY ./README.md ./README.md

# copy the scripts and files you want to use in the container
COPY ./certifications.json ./certifications.json
COPY ./1_download_imagery.py ./1_download_imagery.py
COPY ./2_extract_shorelines.py ./2_extract_shorelines.py
COPY ./3_zoo_workflow.py ./3_zoo_workflow.py
COPY ./5_zoo_workflow_local_model.py ./5_zoo_workflow_local_model.py
COPY ./6_zoo_workflow_with_coregistration.py ./6_zoo_workflow_with_coregistration.py

# copy the pyproject.toml and pixi.lock files
COPY ./pyproject.toml ./pyproject.toml
COPY ./pixi.lock ./pixi.lock

# copy the the notebooks
COPY ./SDS_coastsat_classifier.ipynb /coastseg/SDS_coastsat_classifier.ipynb
COPY ./SDS_zoo_classifier.ipynb /coastseg/SDS_zoo_classifier.ipynb

# install dependencies to `/coastseg/.pixi/envs/`
# use `--locked` to ensure the lockfile is up to date with pixi.toml
# use `--frozen` install the environment as defined in the lock file, doesn't update pixi.lock if it isn't up-to-date with manifest file
# RUN pixi install --locked 
RUN /usr/local/bin/pixi install --manifest-path pyproject.toml --frozen

# This tells Python to include /coastseg/src (where your coastseg package likely resides) when searching for modules.
ENV PYTHONPATH=/coastseg/src:$PYTHONPATH


# Indicate that Jupyter Lab inside the container will be listening on port 8888.
EXPOSE 8888
