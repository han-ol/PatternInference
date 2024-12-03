# Declare phony targets
.PHONY: init-guild setup-env install-hooks update-dependencies clean all

# Default target
all: chmod-scripts init-guild setup-env install-hooks update-dependencies

# Ensure the scripts are executable
chmod-scripts:
	@chmod +x init-guild.sh setup-env.sh install-hooks.sh update-dependencies.sh clean-env.sh
	@echo "Scripts made executable."

# Initialize guild
init-guild:
	@echo "Initializing guild..."
	@./init-guild.sh

# Setup Python environment
setup-env:
	@echo "Setting up Python environment..."
	@./setup-env.sh $(ENV_NAME)

# Install pre-commit hooks
install-hooks:
	@echo "Installing pre-commit hooks..."
	@./install-hooks.sh

# Update and install dependencies
update-dependencies:
	@echo "Updating and installing dependencies..."
	@./update-dependencies.sh

# Clean the environment
clean:
	@echo "Cleaning the project environment..."
	@./clean-env.sh
