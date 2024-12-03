#!/bin/bash

ENV_NAME=$(cat .env_name)

# Ensure conda is initialized
eval "$(conda shell.bash hook)"

if [ -f .env_name ]; then
  ENV_NAME=$(cat .env_name)
  if conda env list | grep -q "^${ENV_NAME}\s"; then
    echo "Deactivating and removing conda environment: ${ENV_NAME}"
    conda deactivate
    conda remove -n ${ENV_NAME} --all -y
    echo "Removing '.env_name' file."
    rm .env_name
  else
    echo "Conda environment ${ENV_NAME} not found."
  fi
else
  echo ".env_name file not found. Cannot determine environment name."
fi

if [ -d .guild ]; then
  echo "Found a .guild directory!"
  echo "If you are sure about removing it, do it yourself with 'rm -rf .guild'"
fi
