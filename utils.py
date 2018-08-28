import re
from collections import Counter
import nltk
from nltk.stem.snowball import SnowballStemmer
import itertools
import pandas as pd


#locate acronyms up to 5 letters in length

def findAcronyms(text):

	acronyms = [re.findall(r'\s\w+\s\w+\s\w+\s\w+\s\([A-Z]+\)', text)]
	acronyms = list(itertools.chain.from_iterable(acronyms))
	acronyms = list(set(acronyms))

	stopwords = nltk.corpus.stopwords.words('english')
	acReduce = [[word for word in acronym.split() if word not in stopwords] for acronym in acronyms]
	acReduce = [' '.join(item) for item in acReduce]


	acronymClean = []
	acronymClean.extend([m.group(0) for m in [re.search(r'\w+\s\w+\s\([A-Z]{2}\)', item) for item in acReduce] if m])
	acronymClean.extend([m.group(0) for m in [re.search(r'\w+\s\w+\s\w+\s\([A-Z]{3}\)', item) for item in acReduce] if m])
	acronymClean.extend([m.group(0) for m in [re.search(r'\w+\s\w+\s\w+\s\w+\s\([A-Z]{4}\)', item) for item in acReduce] if m])
	acronymClean.extend([m.group(0) for m in [re.search(r'\w+\s\w+\s\w+\s\w+\s\([A-Z]{5}\)', item) for item in acReduce] if m])
	
	acronymClean = list(set(acronymClean))
	acronymClean = [item.replace(')', '').split('(')[::-1] for item in acronymClean]
	
	return acronymClean

	
#function to replace acronyms
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

	
def remove_non_ascii(text):
	return ''.join(i for i in text if ord(i)<128)	

	
def preprocess(text):

    # Replace publisher text, other weird stuff
	text = text.lower()
	text = re.sub('\s\d\d\d\d\s', '', text)
	text = text.replace('american chemical society', '')
	text = text.replace('american society of anesthesiologists', '')
	text = text.replace('japanese society of pharmacognosy', '')
	text = text.replace('springer japan', '')
	text = text.replace('royal society of chemistry', '')
	text = text.replace('springer', '')
	text = text.replace('springer-verlag', '')
	text = text.replace('springer verlag', '')
	text = text.replace('elsevier', '')
	text = text.replace('wiley-vch', '')
	text = text.replace('springer science+business', ' ')
	text = text.replace('macmillan publishers limited', ' ')
	text = text.replace('american association for the advancement of science', ' ')
	text = text.replace('springer international publishing', ' ')
	text = text.replace('all rights reserved', ' ')
	text = text.replace('2d', 'two-dimensional')
	text = text.replace('3d', 'three-dimensional')
	text = text.replace('ieee', ' ')
	text = text.replace('acm', ' ')
	text = text.replace('%', ' ')
	text = text.replace('°c', ' ')
	text = text.replace('ltd', ' ')
	text = text.replace('gmbh', ' ')
	text = text.replace('kgaa', ' ')
	text = re.sub('\-\s', ' ', text)
	text = re.sub('\s\-', ' ', text)
	text = text.replace('©', ' ')
	text = text.replace('$', ' ')
	text = re.sub('\d', ' ', text)
	text = re.sub('\s\w\w\s', ' ', text)
	text = re.sub('\s\w\s', ' ', text)
	text = re.sub('\s\W\s', ' ', text)
	text = text.replace('/', ' ')
	text = text.replace(':', ' ')
	text = text.replace('@', ' ')
	text = text.replace('Ã—', ' ')
	text = text.replace(' use ', ' ')
	text = text.replace(' final ', ' ')
	
	return text

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: A list where each item is a tuple of (batch of input, batch of target).
    """
    n_batches = int(len(int_text) / (batch_size * seq_length))

    # Drop the last few characters to make only full batches
    xdata = np.array(int_text[: n_batches * batch_size * seq_length])
    ydata = np.array(int_text[1: n_batches * batch_size * seq_length + 1])

    x_batches = np.split(xdata.reshape(batch_size, -1), n_batches, 1)
    y_batches = np.split(ydata.reshape(batch_size, -1), n_batches, 1)

    return list(zip(x_batches, y_batches))


def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: A tuple of dicts.  The first dict....
    """
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

def stopwords():
	stopwords = nltk.corpus.stopwords.words('english')
	stopwords.extend(('none', 'elsevier', 'ltd', 'springer', 'verlag', 'gmbh', '(ev)', 'rights', 'reserved', 'wiley-vch', 'wiley', '<quotation_mark>', 'royal society', 'royal', 'society', 'also', 'however', 'may', 'ieee', 'publisher', 'conclusion', 'nps', 'one', 'co', 'kgaa', 'kg-1', 'h', 'pv', 'materials', 'device', 'compared', 'obtained', 'based','study', 'publishing', 'analysis', 'appli', 'base', 'accuraci' 'g-1', 'mah','known', 'even', 'form', 'state-of-the-art', 'techniques', 'technique', 'order','different', 'propose', 'compare', 'studies', 'required', 'variable', 'evaluated','use', 'american', 'societi', 'differ', 'signific', 'promis', 'enhanc', 'compar', 'show', 'develop', 'provid', 'improv', 'report', 'low', 'method', 'exhibit', 'addit', 'time', 'prepar', 'rate', 'effect', 'associ', 'scienc', 'poor', 'demonstr', 'feasibl', 'associ advanc', 'advanc scienc', 'area', 'achiev', 'could', 'potenti', 'date', 'present', 'small', 'reach', 'toward', '<semicolon>', '<exclamation_mark>', '<hyphens>', '<left_paren>', '<right_paren>', '<colon>', '<double-quote>', '<copyright>', '<percent>', '<period>', '<comma>', '<question_mark>', 'use', 'observed', 'nm', 'pv', 'used', 'compared', 'high', 'proposed', 'results', 'module', 'b.v.', 'new', 'respectively', 'material', 'properties', 'found', 'due', 'well', 'using', 'investigated', 'image', 'images', 'two', 'data', 'approach', 'parameters', 'methods', 'large', 'paper', 'mppt', '(pv)', 'via', 'ltd.', 'novel', 'taylor', '&', 'figure', 'available', 'inc', 'berlin', 'springer-verlag', 'i.e.', '<dollar>', 'well-known', 'whose', 'without', 'within', 'would', 'thus', 'therefore', 'though', '&co', 'publish'))
			
	return stopwords
	
def tokenize_and_stem(text):

	stemmer = SnowballStemmer('english')
	tokens = [word for word in text.lower().split() if word not in stopwords]
	filtered_tokens = []
	for token in tokens:
		if re.search('[a-zA-Z]', token):
			filtered_tokens.append(token)
	stems = [stemmer.stem(t) for t in filtered_tokens]
	return stems

def tokenize_only(text):

	tokens = [word for word in text.lower().split() if word not in stopwords]
	filtered_tokens = []
	for token in tokens:
		if re.search('[a-zA-Z]', token):
			filtered_tokens.append(token)
	return filtered_tokens

stopwords = stopwords()