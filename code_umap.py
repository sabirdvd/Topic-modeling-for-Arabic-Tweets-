import pandas as pd
from bertopic import BERTopic
from flair.embeddings import TransformerDocumentEmbeddings
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
from gensim.models import LdaMulticore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from umap import UMAP

#data=  pd.read_csv("arabic_dataset_classifiction_reduit.csv")
''''
 We used "DataSet for Arabic Classification" [1] which contains 111,728 Arabic documents written in Modern
Standard Arabic (MSA). The dataset was collected from three Arabic online newspapers: Assabah, Hespress and
Akhbarona. The documents in the dataset are categorized into 5 classes: sport, politics, culture, economy and diverse.
We removed 2939 missing documents and ran the experiments with the remaining 108789 documents without any
document labels. The text contains only alphabetic, numeric and symbolic words, so we have not applied any
preprocessing as the text is almost clean
[1] M. Biniz, “DataSet for Arabic Classification,” vol. 1, Mar. 2018, doi: 10.17632/v524p5dhpj.1
'''
  
  
#arabic_dataset_classifiction
data = pd.read_csv("arabic_dataset_classifiction.csv")
# tweets results 
#('عناصر', 0.005736651291083146), 
#('المتهم', 0.005521592810937796), 
#('الدرك', 0.00502498458456131), 
#('القضائية', 0.0050212182104924), 
#('اعتقال', 0.004773258920235465), 
#('المخدرات', 0.004162502451645349), 
#('الابتدائية', 0.00396287623139702), 
#('النيابة', 0.0039598014028783286), 
#('الشرطة', 0.003955431150969441), 
#('الضابطة', 0.003934071447302295)

# results 
#data = pd.read_csv("tweet_print_v2.csv")
#data = pd.read_csv("SaudiIrony.csv")
data.head()

# shape  
data.shape
data = data.dropna()
documents = data['text'].values
arabert = TransformerDocumentEmbeddings('aubmindlab/bert-base-arabertv02')

# Topic Modeling
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
topic_model = BERTopic(language="arabic", low_memory=True ,calculate_probabilities=False, embedding_model=arabert, umap_model=umap_model)
topics, probs = topic_model.fit_transform(documents)

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
cm = CoherenceModel(topics=topics, texts=texts, corpus=corpus, dictionary=id2word, coherence='c_npmi')
coherence = cm.get_coherence() 
print('\nCoherence Score: ', coherence)


# Visualize the topics
topic_model.visualize_topics()

# save the twitter 
topic_model.save("model_ump")	
# Load model
model = BERTopic.load("model_ump")
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


