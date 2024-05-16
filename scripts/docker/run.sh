#!/usr/bin/env bash

docker build -t chatgpt-local .
docker run -it chatgpt-local