# Topic modeling for Arabic Tweets


Create conda  environments

```
conda create -n AraTop  python=3.7 anaconda 
conda activate AraTop   
``` 


Install req 
```
pip install bertopic 
pip install flair  
``` 

# Dataset 
The dataset is based on ArabGend dataset 2022 [1] 108053 tweets 

Getting the twitt ID from [1] and then retrive tweets using [Twitter API](https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api) 

```
pip install twarc
twarc2 hydrate ids.txt tweets.json
twarc2 hydrate twitt_ID.txt tweets.json
``` 

Convert json file to CSV [twarc](https://github.com/DocNow/twarc-csv)

```
pip3 install --upgrade twarc-csv
twarc2 csv --no-json-encode-all tweets.json tweets_CSV.csv
csvcut --columns id,text tweets_CSV.csv
```







To clean and pre-process the dataset 

```
python arabic_cleaner.py
```

clean dataset 
``` 
wget https://www.dropbox.com/s/3fqno4wxyktlclu/cleaned_tweet_gen_remove_emoji_v4.csv.zip 
unzip cleaned_tweet_gen_remove_emoji_v4.csv.zip
```

[1] [ArabGend:Gender Analysis and Inference on Arabic Twitter](https://aclanthology.org/2022.wnut-1.14.pdf)



# Training

For Topic modeling via [umap](https://umap-learn.readthedocs.io/en/latest/basic_usage.html)

```
run_umap.sh
```

For Topic modeling via [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)

```
run_hdbscan.sh
``` 

For joint model (umap+hdbscan)

```
run joint.sh 
```


# Infereance 

loading the tranined model 

```
python infer.py
``` 


