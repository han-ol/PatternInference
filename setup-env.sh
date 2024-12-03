#!/bin/bash

ENV_NAME=$1

if [ -z "$ENV_NAME" ]; then
  echo "Usage: $0 <env-name>"
  exit 1
fi

# Save the environment name to a file
echo "$ENV_NAME" > .env_name

if conda env list | grep -q "^${ENV_NAME}\s"; then
  echo "The environment '$ENV_NAME' already exists."
else
  conda create -n "$ENV_NAME" python=3.10 -y
fi

echo "Activating Python environment: $ENV_NAME"
# Ensure conda is initialized
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Check if the environment was successfully activated
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
  echo "Failed to activate the environment '$ENV_NAME'. Exiting..."
  exit 1
fi

echo "Environment '$ENV_NAME' activated successfully."

echo "Installing dependencies..."
pip install -r requirements.txt
pip install pre-commit
pip install pip-tools
pip install -e .

echo "Environment setup complete."
