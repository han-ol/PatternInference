#!/bin/bash

# Set the project path to the directory where this script is located
PROJECT_PATH=$(dirname "$0")

# Check if the .guild directory already exists
if [ -d "$PROJECT_PATH/.guild" ]; then
  echo "The .guild directory already exists."
  echo "Show past runs by running 'guild compare'."
  echo "Please run 'guild help' to see available operations."
  exit 1
else
  # Create the .guild directory
  mkdir "$PROJECT_PATH/.guild"
  echo ".guild directory created successfully."
fi
