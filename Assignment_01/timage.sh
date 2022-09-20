#!/bin/bash

# download and unzip dataset
#wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip

current="$(pwd)/tiny-imagenet-200"
dest="/home/dhruvb/adrl/datasets/tiny_imagenet/"
mkdir $dest
# training data
cd $current/train
for DIR in $(ls); do
   cd $DIR
   rm *.txt
   mv images/* $dest
   rm -r images
   cd ..
done

cd $current/val
for DIR in $(ls); do
   cd $DIR
   rm *.txt
   mv images/* $dest
   rm -r images
   cd ..
done

cd $current/test
for DIR in $(ls); do
   cd $DIR
   rm *.txt
   mv images/* $dest
   rm -r images
   cd ..
done

echo "done"