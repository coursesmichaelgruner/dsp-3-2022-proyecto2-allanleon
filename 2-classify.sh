#!/bin/bash

# The correct ones are not going to be altered

# Merge non-background noise to unknown samples
mkdir data/unknown

UNKNOWN="bed bird cat dog eight five four happy house marvin nine one seven sheila six three tree two wow zero"

# Move the samples and alter the lists
echo "Classifying samples in new classes"
cd data
for i in $UNKNOWN; do
  mv $i/* unknown/
  rmdir $i
done

# Move the background noise
echo "Moving background noise to a folder without underscores"
mv _background_noise_ background_noise

# Counting classes
CLASSES="background_noise down go left no off on right stop unknown up yes"
for i in $CLASSES; do
  echo "$i: $(ls $i | wc -l)"
done
cd -
