
# Instrucciones

# Dependencies


```bash
sudo apt install mpich libopenmpi-dev graphviz -y
```

## Python modules for spectrogram:

```bash
pip3 install librosa numpy matplotlib mpi4py pandas oct2py pydot
```

# Ejecutar

```bash
git clone git@github.com:coursesmichaelgruner/dsp-3-2022-proyecto2-allanleon.git
cd dsp-3-2022-proyecto2-allanleon

# Download datasets and compute spectrograms
./1-init.sh
# If you already have the speech_commands_v0.01.tar.gz tar,
# you can create a folder called data and extract it there
mkdir data
tar -xvzf speech_commands_v0.01.tar.gz -C ./data
#

#Generate noise clips
./6-python-clip.py

# Classify unknown commands
./2-classify.sh

# Spilt into training and validation samples
./3-split-dataset.sh

# Compute spectograms (takes around 5 mins)
mpirun -np 5 --use-hwthread-cpus python3 4-compute-spectrogram.py data/ 2> err.log

# Visualize spectograms
./5-python-visualise.py

# Visualize label distribution
./7-sample-dist.py data/

# Train the model
./train.py data

#Validate model  (this also computes CPU and spectogram throughput)
./validate.py data

# Predict own input
./predict.py <wav file to predit or folder with several files>
e.g.: ./predict.py muestras/down.wav # should display a message with 'down'
```

For single view of spectogram:

```bash
python3 4-compute-spectrogram.py -s data/training/up/9785931e_nohash_0.wav
```

The `-np` must be multiple of 5
