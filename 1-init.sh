#!/bin/bash

mkdir data
echo "Downloading data"
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz > /dev/null
mv speech_commands_v0.01.tar.gz data

echo "Decompressing"
cd data && tar xzf speech_commands_v0.01.tar.gz && cd -
