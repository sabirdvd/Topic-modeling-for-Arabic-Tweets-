# Topic modeling for Arabic Tweets

This code is adopted from this study [BERT for Arabic Topic Modeling: An Experimental Study on BERTopic Technique](https://www.sciencedirect.com/science/article/pii/S1877050921012199) [(code)](https://github.com/iwan-rg/Arabic-Topic-Modeling)


Please refer to this [blog post](https://ahmed.jp/blog/2022-12-ArabTop/ArabTop_2022.html) for more detail about this repository. 

[Interactive demo](https://ahmed.jp/visual_topics.html)


## Table of Contents
- <a href='#Dataset'>Dataset</a>
- <a href='#Training'>Training</a>
- <a href='#Inference '>Inference</a>



<!--
- <a href='#Image-Text-Retrieval'>Image/Text Retrieval</a>
-->


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

## Dataset 
The dataset is based on the ArabGend dataset 2022 [1] 108053 tweets  

Getting the tweets ID from data file or from [1] or and then [retrieve tweets](https://twittercommunity.com/t/arabic-tweets-in-unicode/159595/2) using [Twitter API](https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api) 

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
<!--
Clean dataset 
``` 
wget https://www.dropbox.com/s/87hnyoi8ep073gh/arab_gen_twitter.csv.zip
unzip arab_gen_twitter.csv.zip
```
-->

[1] [ArabGend:Gender Analysis and Inference on Arabic Twitter](https://aclanthology.org/2022.wnut-1.14.pdf)



## Training

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

## Inference 

loading the tranined model 

```
python infer.py
``` 
 
## Citation

If you find this blog useful, please kindly cite it:

```bibtex
@article{sabir2022arabtop,
  title   = "Review: Topic Modeling for Arabic Language",
  author  = "Sabir, Ahmed",
  year    = "2022",
  url     = "https://ahmed.jp/blog/2022-12-ArabTop/ArabTop_2022"
}
```

