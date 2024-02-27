#!/bin/bash

# Assume this is ran from the main fusion-idlab directory
dir="./models/"

filename="fusion_idlab_ckpt7.tar.gz"
download_url="https://cloud.ilabt.imec.be/index.php/s/dWL4AkkeNmqxPPe/download/fusion_idlab_ckpt7.tar.gz"
wget $download_url -P $dir

mkdir -p $dir
cd $dir
tar -xzf $filename