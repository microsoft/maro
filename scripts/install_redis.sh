#!/bin/bash

# script to install Redis at current direction on linux.

if [ $(uname) != "Darwin" ]
then
  wget http://download.redis.io/releases/redis-6.0.6.tar.gz
  tar xzf redis-6.0.6.tar.gz
  cd redis-6.0.6
  make
fi
