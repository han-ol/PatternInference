#!/bin/bash

echo "Attempting to activate environment to ensure 'pre-commit' is available."
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


pre-commit install
echo "Pre-commit hooks installed."
