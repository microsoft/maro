#!/bin/bash
if python3 -c 'import maro' > /dev/null 2>&1; then
  # maro module is already installed, skip
  exit 0
else
  # Install maro
  wget https://github.com/J-shang/maro/archive/v0.2_tuner.zip
  sudo apt install unzip
  unzip v0.2_tuner.zip
  cd maro-0.2_tuner
  bash scripts/install_maro.sh
  sudo apt update
  sudo apt install -y redis-tools
fi