#!/bin/bash

echo "Downloading data"
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz > /dev/null
mv speech_commands_v0.01.tar.gz data

echo "Decompressing"
mkdir data
cd data && tar xzf speech_commands_v0.01.tar.gz && cd -
