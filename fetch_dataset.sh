#!/bin/bash

wget https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz
tar -xvzf nf_prize_dataset.tar.gz
mv download/ data/
rm nf_prize_dataset.tar.gz
