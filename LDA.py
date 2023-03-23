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


data = pd.read_csv("arab_gen_twitter.csv")
data.head()
# shape
data.shape
data = data.dropna()   
   
documents = data['text'].values

exts = [[word for word in str(document).split()] for document in documents]
id2word = corpora.Dictionary(texts)
corpus = [id2word.doc2bow(text) for text in texts]   
   
no_topics = 5
lda = LdaMulticore(corpus, id2word=id2word, num_topics=no_topics)
#compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda, texts=texts, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)   


