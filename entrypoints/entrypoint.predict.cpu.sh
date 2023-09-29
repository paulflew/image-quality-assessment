#!/bin/bash
set -e

IMAGE_SOURCE=$1

# predict
python3 -m evaluater.predict -w /src/weights -is $IMAGE_SOURCE
