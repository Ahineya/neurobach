#!/bin/bash

if [ -z "$1" ]
  then
    echo "Usage: ./train.sh model_name"
    exit 1
fi

python3 trainer.py train ./output/output.txt --model-name $1