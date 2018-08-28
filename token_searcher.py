import pandas as pd
import itertools
import nltk
import sys

import utils

from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation
from sklearn.manifold import TSNE
from nltk.text import TokenSearcher

from gensim.models import Phrases
from gensim.models.phrases import Phraser


'''
NLTK TokenSearcher to ID abstracts containing desired words; change the regex text after findall according to what you want to search for. 
Try using the most_similar function or the tsne_plot_word function to identify words in a group.
This creates a list of two lists, the abstract tokens you created above and the words found using findall.
'''

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
absCl = list(map(lambda x: utils.remove_non_ascii(x), absCl))

stopwords = nltk.corpus.stopwords.words('english')

texts = [[word for word in abstract.lower().split() if word not in stopwords] for abstract in absCl]

phrases = Phrases(texts)
bigram = Phraser(phrases)
trigram = Phrases(bigram[texts])
trigram = Phraser(trigram)
texts = [trigram[bigram[text]] for text in texts]

y = []
i = 0
while i < len(texts):
	topics = TokenSearcher(nltk.Text(texts[i])).findall(r'<.*addict.*|opioid_use|.*dependence.*|.*abuse.*|.*abuse|.*alcoholi.*|.*inject_drugs|people_inject.*|drugs_people|.*sober.*|.*misuse.*|.*detox.*|.*heroin.*|hepatitis|.*illicit.*|.*overdose.*|drug_use|drug_use.*|substance_use|treatment_facility|recovering.*> <.*>')
	if topics:
		y.append([texts[i], topics[:], 'Addiction/Abuse'])
	elif not topics:
		topics = TokenSearcher(nltk.Text(texts[i])).findall(r'<anesthe.*|.*anesthe.*|.*anesthe|icu|.*perioper.*|.*arthroplasti.*|.*postop.*|.*inpatient.*|.*outpatient.*|sevoflurane|midazolam|.*epidural.*|ropivacaine|.*cancer.*|.*surgic.*|.*surger.*|.*cesarean.*|.*caesarean.*|.*lymphoma.*|.*laparoscop.*|dexmedetomidin|.*sedat.*|.*operat.*|.*endoscop.*|.*radiolo.*|.*paracetamol.*> <.*>')
		if topics:
			y.append([texts[i], topics[:], 'Medical Procedure'])
		elif not topics:
			topics = TokenSearcher(nltk.Text(texts[i])).findall(r'<.*pain.*|acetaminophen|.*analgesic.*|.*analgesi.*|ropivacain|.*antinocicept.*|.*nocicep.*|.*inflamm.*|.*epidural.*|.*formalin.*|.*fentanyl.*|oxycodone|remifentanil|.*hyperalgesia.*|nerve_block.*|gabapentin|kappa_opioid|pallative_care|.*paracetamol.*> <.*>')
			if topics:
				y.append([texts[i], topics[:], 'Pain Management'])
			elif not topics:
				topics = TokenSearcher(nltk.Text(texts[i])).findall(r'<.*methadone.*|.*treatment.*|.*therap.*|.*delivery.*|.*transport.*> <.*>')
				if topics:
					y.append([texts[i], topics[:], 'Methadone | Treatment/Therapy | Delivery'])
				elif not topics:
					topics = TokenSearcher(nltk.Text(texts[i])).findall(r'<autis.*|.*parkins.*|.*menorrh.*|.*child.*|.*pregnan.*|.*neonat.*|.*palliat.*|.*posttraum.*|.*disorder.*|.*syndro.*|.*traum.*|.*disease> <.*>')
					if topics:
						y.append([texts[i], topics[:], 'Conditions or disorders'])
					elif not topics:
						topics = TokenSearcher(nltk.Text(texts[i])).findall(r'<xxnonexx>')
						if topics:
							y.append([texts[i], topics[:], 'No Abstract'])
						elif not topics:
							topics = TokenSearcher(nltk.Text(texts[i])).findall(r'<.*morphi.*|.*dependent.*|.*prescrip.*|.*prescrib.*|.*opioid.*> <.*>')
							y.append([texts[i], topics[:], 'Misc'])
	i += 1

#converts the list y into small dataframe, merges with original dataframe
#NOTE: remove undesired columns from original df (scopus provides a lot of blank garbage)

columns = ['tokenized abst text', 'found topics', 'simple subject']
dfNew = pd.DataFrame(y, columns = columns)
dfCombo = pd.concat([df, dfNew], axis = 1)

writer = pd.ExcelWriter('tokens_found.xlsx', engine = 'xlsxwriter')
dfCombo.to_excel(writer, encoding ='ISO-8859-1')
writer.save()

dfAddict = dfCombo.loc[dfCombo['simple subject'] == 'Addiction/Abuse']
dfMed = dfCombo.loc[dfCombo['simple subject'] == 'Medical Procedure']
dfPain = dfCombo.loc[dfCombo['simple subject'] == 'Pain Management']
dfTreat = dfCombo.loc[dfCombo['simple subject'] == 'Methadone | Treatment/Therapy | Delivery']
dfCond = dfCombo.loc[dfCombo['simple subject'] == 'Conditions or disorders']
dfMisc = dfCombo.loc[dfCombo['simple subject'] == 'Misc']