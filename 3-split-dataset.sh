#!/bin/bash

# Determine the distribution of the validation list
# We are aware that there is a mean value of 2350 samples. So, for validation,
# we should use something like 10% (235 samples)

cd data

CLASSES="down go left no off on right stop unknown up yes"
mkdir validation
mkdir training

echo "Splitting randomly into validation and training dirs"
for i in $CLASSES; do
  mkdir validation/$i
  mkdir training/$i
  cd $i
  echo "$i total: $(ls | wc -l) files"
  shuf -zn235 -e *.wav | xargs -0 mv -t ../validation/$i/
  cd -
  mv $i/* training/$i/
  echo "$i training: $(ls training/$i/ | wc -l) files"
  echo "$i validation: $(ls validation/$i/ | wc -l) files"
  rmdir $i
done

echo "Building validation and training list files"
# Clear existing ones
rm -f validation_list.txt
rm -f training_list.txt
rm -f testing_list.txt

# Recreate
for i in $CLASSES; do
  ls -1 validation/$i/* >> validation_list.txt
  ls -1 training/$i/* >> training_list.txt
done

cd -
