import pandas as pd
from bertopic import BERTopic
from flair.embeddings import TransformerDocumentEmbeddings
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
from gensim.models import LdaMulticore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

#data = pd.read_csv("tweet_print_v2.csv")
#data = pd.read_csv("/Users/asabir/Downloads/ahmad_test/tweet_gen_remove_emoji.csv_cleaned.csv")
data = pd.read_csv("cleaned_tweet_gen_remove_emoji_v4.csv")
#data = pd.read_csv("SaudiIrony.csv")
data.head()

# shape  
data.shape
data = data.dropna()
documents = data['text'].values
arabert = TransformerDocumentEmbeddings('aubmindlab/bert-large-arabertv02-twitter')

# Topic Modeling
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

model = BERTopic(language="arabic", low_memory=True ,calculate_probabilities=False, embedding_model=arabert, hdbscan_model=hdbscan_model, umap_model=umap_model)

#umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
#model = BERTopic(language="arabic", low_memory=True ,calculate_probabilities=False, embedding_model=arabert, umap_model=umap_model)


topics, probs = model.fit_transform(documents)

#extract most frequent topics
topic_model.get_topic_freq().head(5)                     
topic_model.get_topic(1)[:10]

texts = [[word for word in str(document).split()] for document in documents]
id2word = corpora.Dictionary(texts)
corpus = [id2word.doc2bow(text) for text in texts]

topics=[]
for i in topic_model.get_topics():
  row=[]
  topic= topic_model.get_topic(i)
  for word in topic:
     row.append(word[0])
  topics.append(row)


# compute coherence score
#cm = CoherenceModel(topics=topics, texts=texts, corpus=corpus, dictionary=id2word, coherence='c_npmi')
#coherence = cm.get_coherence() 
#print('\nCoherence Score: ', coherence)


# Visualize the topics
#topic_model.visualize_topics()

# save the twitter 
topic_model.save("model_twitter_joint")	
# Load model
model = BERTopic.load("model_twitter_joint")
#chang the number of topics here
no_topics = 5

# run LDA 
lda = LdaMulticore(corpus, id2word=id2word, num_topics=no_topics)
#compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda, texts=texts, dictionary=id2word, coherence='c_npmi')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# run NMF
# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


#chang the number of topics here
no_topics = 5

# run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

topics_NMF=[]
for index, topic in enumerate(nmf.components_):
    row=[]
    for i in topic.argsort()[-10:]:
      row.append(tfidf_vectorizer.get_feature_names()[i])
    topics_NMF.append(row)


cm = CoherenceModel(topics=topics_NMF, texts=texts, corpus=corpus, dictionary=id2word, coherence='c_npmi')
coherence_nmf = cm.get_coherence()  
print('\nCoherence Score: ', coherence_nmf)


