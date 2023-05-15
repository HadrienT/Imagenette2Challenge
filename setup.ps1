# This is a comment in PowerShell
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\Activate

# Install required packages
pip install -r requirements.txt
pip install -r requirements_dev.txt

# Install the current directory as a package
pip install -e .
