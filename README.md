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
The dataset is based on ArabGend dataset [1] 



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



# Code 

```
run.sh
```




