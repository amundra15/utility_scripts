#!/bin/bash

# Define the source and destination paths
source_script="path_converter.py"
destination_directory="$HOME"
alias_file="$HOME/.bash_aliases"

# Copy the script to the home directory
cp "$source_script" "$destination_directory"

# Add aliases to .bash_aliases
echo 'alias w2l="python3 $HOME/path_converter.py w2l"' >> "$alias_file"
echo 'alias l2w="python3 $HOME/path_converter.py l2w"' >> "$alias_file"

echo "Enter 'source "$HOME/.bashrc"' in command line to update the aliases."
