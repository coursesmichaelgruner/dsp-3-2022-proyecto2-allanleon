
#Instrucciones

Ejecutar

```bash
./1-init.sh
./2-classify.sh
./3-split-dataset.sh
mpirun -np 5 --use-hwthread-cpus python3 4-compute-spectrogram.py data/ 2> err.log
```

For single view:

```bash
python3 4-compute-spectrogram.py -s data/training/up/9785931e_nohash_0.wav
```

The `-np` must be multiple of 5

Install


MPI for task farming:

```bash
sudo apt install mpich libopenmpi-dev -y
```

Python modules for spectrogram:

```bash
pip3 install librosa numpy matplotlib
pip3 mpi4py
```

