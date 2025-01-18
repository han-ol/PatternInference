#!/bin/bash

echo "Attempting to activate environment before updating dependencies."
ENV_NAME=$(cat .env_name)

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

echo "Updating dependencies with pip-compile..."
pip-compile --pip-args "--extra-index-url https://download.pytorch.org/whl/cpu"
echo "Dependencies updated. Installing updated requirements..."
pip-sync
echo "ADDITIONALLY INSTALLING BAYESFLOW FROM dev BRANCH !!! even though bayesflow is not mentioned in pyproject.toml or requirements.txt"
pip install git+https://github.com/bayesflow-org/bayesflow.git@3f82ce1
echo "Dependencies installed."
