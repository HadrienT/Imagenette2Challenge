#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

# Update system packages
sudo apt update
sudo apt upgrade -y

# Install necessary packages
sudo apt install -y git python3-pip python3-venv

# Clone the Git repository
git clone https://github.com/HadrienT/Imagenette2Challenge.git

# Change to the cloned repository directory
cd Imagenette2Challenge

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --no-cache
pip install -e . --no-cache
