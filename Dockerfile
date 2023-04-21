# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Create a directory for the external hard drive data
RUN mkdir /data

# Set the working directory
WORKDIR /docker_root

# Copy the requirements files into the container
COPY requirements.txt requirements_dev.txt setup.py pyproject.toml setup.cfg ./

# Copy the rest of the application code into the container
COPY src ./src
COPY tests ./tests

# Install any needed packages specified in requirements.txt and requirements_dev.txt
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -e . --no-cache-dir

# Set an environment variable for the data directory
ENV DATA_DIR /data

