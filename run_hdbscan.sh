#!/bin/bash
export CONDA_ALWAYS_YES="true"
conda create -n AraTop_hdbscan python=3.7 anaconda &&
conda activate AraTop_hdbscan  
pip install bertopic &&
pip install flair  &&
wget https://www.dropbox.com/s/3fqno4wxyktlclu/cleaned_tweet_gen_remove_emoji_v4.csv.zip &&
unzip cleaned_tweet_gen_remove_emoji_v4.csv.zip  &&
python code_hdbscan.py && 
command; echo "Process done" | mail -s "Process done" your@email.com 
