#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from mpi4py import MPI
from spectrogram import compute_spectrogram, plot_spectrogram

# MPI specific
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

# For a single job
def single_job(input_file):
    y, fs = librosa.load(input_file)
    S_dB = compute_spectrogram(y, fs)
    print(S_dB.shape)
    plot_spectrogram(S_dB, fs)

# For MPI task farming
def process_files(files):
    total = len(files)
    total_10perc = total / 10 

    for i in range(total):
        # Progress tracker
        if i % 500 == 0:
            print(f"{rank} has completed {i}/{total} files", flush=1)
        
        # Process file
        file = files[i]
        y, fs = librosa.load(file)
        S_dB = compute_spectrogram(y, fs)

        # Save file
        filename = os.path.splitext(os.path.basename(file))[0]
        filename = os.path.join(os.path.dirname(file), filename + '.npy')
        np.save(filename, S_dB)
        

def open_file(file):
    f = open(file, 'r')
    lines = f.readlines()
    return lines

def parse_lists(path):
    # Open both files
    val = os.path.join(path, 'validation_list.txt')
    train = os.path.join(path, 'training_list.txt')
    files = open_file(val) + open_file(train)
    return [os.path.join(path, i.strip('\n')) for i in files]

# Main process
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter,
                                     description="Program to compute the spectrograms")

    parser.add_argument('input', help="filename or path to process")
    parser.add_argument('-s', '--single', action='store_true',
                        help="If not single, training_list.txt and validation_list.txt" +
                        " will be opened to process in task-farming mode")

    args = parser.parse_args()

    # Single job process
    if args.single and rank != 0:
        exit()

    if args.single and rank == 0:
        single_job(args.input)

    # Multi job
    if rank == 0:
        # Master process
        print("Reading files from Validation and Training lists ", flush=1)
        wholelist = parse_lists(args.input)
        print(f"Total files: {len(wholelist)}", flush=1)

        chunks = np.split(np.array(wholelist), world_size)
        files = comm.scatter(chunks, root=0)
    else:
        # Slave processes
        files = comm.scatter(None, root=0)
    
    print(f"Received {len(files)} in process {rank}")
    process_files(files)
