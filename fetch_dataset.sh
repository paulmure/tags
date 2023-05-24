#!/bin/bash

# >>>> Netflix >>>>
wget https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz
tar -xvzf nf_prize_dataset.tar.gz
mkdir -p data/netflix
mv download/* data/netflix/
rmdir download
rm nf_prize_dataset.tar.gz

tar -xvf data/netflix/training_set.tar -C data/netflix
./scripts/transform_netflix_data.py
# <<<< Netflix <<<<
