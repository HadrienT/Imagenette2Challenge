#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

# Update system packages
sudo apt update
sudo apt upgrade -y

# Install necessary packages
sudo apt install -y git docker.io

# Clone the Git repository
git clone https://github.com/HadrienT/Imagenette2Challenge.git


# Change to the cloned repository directory
cd Imagenette2Challenge

# Build the Docker image from the Dockerfile
sudo docker build -t i2c_image .

# Run the Docker container with external SSD mounted as volume
sudo docker run -d --name i2c_container -v /mnt/e:/data i2c_image

sudo docker cp .env i2c_container:Imagenette2Challenge/.env

