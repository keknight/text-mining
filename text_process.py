import itertools
import utils
import gensim
import nltk
import re
import sys

import pandas as pd
import numpy as np

from nltk.stem.snowball import SnowballStemmer
from gensim import corpora, models, similarities, matutils
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from collections import defaultdict


#function to compare probability values of categories 
#to a given value (in this code, the closest value to 1)
#returns index of category
def find_nearest_topic(array, values):
    indices = np.abs(np.subtract.outer(array, values)).argmin(0)
    return indices

#preprocessing function to use gensim built-in stuff
def preprocess(row):
	return strip_punctuation(remove_stopwords(row.lower()))
	
def js_dist(X):
    return pdist(X, lambda u, v: jensen_shannon(u, v))


#read training document data from command line, convert columns to lists
#assumes that file name is passed at CL as last argument

filename = sys.argv[-1]
df = pd.read_excel(filename)


abst = list(itertools.chain.from_iterable(df[['Abstract']].as_matrix().tolist()))
titles = list(itertools.chain.from_iterable(df[['Title']].as_matrix().tolist()))
eids = list(itertools.chain.from_iterable(df[['EID']].as_matrix().tolist()))

#run acronyms and other preprocessing functions on data
acronyms = list(map(lambda x: utils.findAcronyms(x), abst))
acronyms = list(itertools.chain.from_iterable(acronyms))
acroDict = {' ' + a[0] + ' ':' '+ a[1] for a in acronyms}

abst = [strip_punctuation(row) for row in abst]
absCl = list(map(lambda x: utils.replace_all(x, acroDict), abst))
absCl = list(map(lambda x: utils.preprocess(x), absCl))
#absCl = [preprocess(row) for row in absCl]
absCl = list(map(lambda x: utils.remove_non_ascii(x), absCl))

tiCl = list(map(lambda x: utils.replace_all(x, acroDict), titles))
tiCl = list(map(lambda x: utils.preprocess(x), tiCl))
#tiCl = [preprocess(row) for row in tiCl]
tiCl = list(map(lambda x: utils.remove_non_ascii(x), tiCl))

#create stopwords
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(('none', 'ltd', 'gmbh', '(ev)', 'wiley', 'also', 'however', 'may', 'ieee', 'publisher', 'conclusion', 'nps', 'one', 'co', 'kgaa', 'kg-1', 'h', 'pv', 'materials', 'device', 'compared', 'obtained', 'based','study', 'publishing',  'age', 'success', 'event', 'ed', 'inf', 'furthermore', 'suggest', 'cm', 'confirm', 'moreover', 'first', 'challenge'))

#create tokenized lists for both the abstracts and the titles
texts = [[word for word in abstract.lower().split() if word not in stopwords] for abstract in absCl]
tiTexts = [[word for word in title.lower().split() if word not in stopwords] for title in tiCl]

#Stemmed text list using NLTK stemmer
stemmer = SnowballStemmer('english')
tiTexts = [[stemmer.stem(x) for x in word] for word in tiTexts]

#create phrases based on abstract texts, apply stemmer
phrases = Phrases(texts)
bigram = Phraser(phrases)
trigram = Phrases(bigram[texts])
trigram = Phraser(trigram)
phText = [trigram[bigram[text]] for text in texts]

texts = [[stemmer.stem(x) for x in word] for word in phText]

#create dictionary using the title phrases

dictionary = corpora.Dictionary([text for text in texts])

#filter out low occurring words
#save dictionary to directory
dictionary.filter_extremes(no_below=50, no_above = 0.6)
dictionary.save_as_text('abstract_dictionary')

#create a "bag of words" corpus based on abstracts dictionary
#populated by words/phrases from titles to reduce noise
#store corpus to disk
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('abstract_text_corpus.mm', corpus)