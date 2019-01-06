# features set 
# number of Bad/curse/racist words occured (need a set of bad words)
# accumulated offensive polarity (caculated as sum of offensive polariy of each word. Offensive polarity is average number of times a word occured in offensive sentences)
# accumulated non-offensive polarity 
# number of words
# length of text
# sentiment intenisty using vader (4 numerical features)

import numpy as np
import pandas as pd

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.classes.spellcorrect import SpellCorrector

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer

from joblib import dump
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    # annotate={"hashtag", "allcaps", "elongated", "repeated",
    #     'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    # corpus from which the word statistics are going to be used 
    # for spell correction
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

data = pd.read_csv('data/offenseval-training-v1.tsv', sep ='\t',header= 0,encoding='utf-8')

def load_data(data):
  text, label_a, label_b, label_c = [], [], [], []
  data = data.fillna('nothing')
  arr = data.as_matrix()
  text = arr[:,1]
  label_a = arr[:,2]
  label_b = arr[:,3]
  label_c = arr[:,4]
  for i in range(len(label_a)):
    if(label_a[i] == 'NOT'):
      label_a[i] = 0
    elif(label_a[i] == 'OFF'):
      label_a[i] = 1
  for i in range(len(label_b)):
    if(label_b[i] == 'TIN'):
      label_b[i] = 0
    elif(label_b[i] == 'UNT'):
      label_b[i] = 1
    elif(label_b[i] == 'nothing'):
      label_b[i] = 2
  for i in range(len(label_c)):
    if(label_c[i] == 'GRP'):
      label_c[i] = 0
    elif(label_c[i] == 'IND'):
      label_c[i] = 1
    elif(label_c[i] == 'nothing'):
      label_c[i] = 3
    elif(label_c[i] == 'OTH'):
      label_c[i] = 2  
  return text, label_a, label_b, label_c

text, label_a, label_b, label_c = load_data(data)

from nltk.corpus import stopwords
import string
from scipy import sparse
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
import re

def clean_data(corpus):
	new_corp = []
	# spell correct
	for i in range(len(corpus)):
		t = re.sub('@USER', '', corpus[i])
		t = text_processor.pre_process_doc(t)
		t = [sp.correct(word) for word in t if word not in string.punctuation]
		new_corp.append(" ".join(t))
	new_corp2 = []
	# lemmatize
	for i in range(len(corpus)):
		if (i%1000 == 0):
			print (i) # for logging purpose
		t = new_corp[i].split(" ")
		to_add = []
		for i in t:
			if i not in stop_words:
				to_add.append(wordnet_lemmatizer.lemmatize(i))
		new_corp2.append(" ".join(to_add))
	del new_corp
	return new_corp2

sp = SpellCorrector(corpus="english") 
wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

lemmatized_text = clean_data(text)

def build_offens_and_non_offens_polarity(corpus, offens_label):
	off_words = {}
	non_off_words = {}
	# count occurence of words in offensive and non-offensive sentence
	for i in range(len(offens_label)):
		print(i, corpus[i])
		if offens_label[i] == 1:
			t = corpus[i].split(" ")
			for word in t:
				if not word in off_words:
					off_words[word] = 0
				off_words[word] += 1
		else:
			t = corpus[i].split(" ")
			for word in t:
				if not word in non_off_words:
					non_off_words[word] = 0
				non_off_words[word] += 1
	
	# normalize count
	mean = 0
	for word in off_words:
		mean+=off_words[word]
	mean = mean/len(off_words)

	sde = 0
	for word in off_words:
		sde += (off_words[word] - mean)**2
	sde = (sde/len(off_words))**0.5

	for word in off_words:
		off_words[word] = (off_words[word] - mean)/sde
	
	mean = 0
	for word in non_off_words:
		mean+=non_off_words[word]
	mean = mean/len(non_off_words)

	sde = 0
	for word in non_off_words:
		sde += (non_off_words[word] - mean)**2
	sde = (sde/len(non_off_words))**0.5

	for word in non_off_words:
		non_off_words[word] = (non_off_words[word] - mean)/sde

	return off_words, non_off_words

def load_bad_words():
	bad_words = set()
	f = open('data/bad-words.txt')
	bad_list = f.read().splitlines() 
	for word in bad_list:
		bad_words.add(word)
	return bad_words

train_txt, test_txt, train_lemtxt, test_lemtxt, train_lab, test_lab  = train_test_split(text, lemmatized_text, label_a, test_size = 0.2, random_state=42)

col_names = ['text', 'lemm_text', 'off_label']

df_train = pd.DataFrame({'text':train_txt, 'lemm_text':train_lemtxt, 'off_label':train_lab}, columns = col_names)
df_test = pd.DataFrame({'text':test_txt, 'lemm_text':test_lemtxt, 'off_label':test_lab}, columns = col_names)

df_train.to_csv('data/train.csv', index=False, sep='\t')
df_test.to_csv('data/test.csv', index=False, sep='\t')

# print(df_train)
# df_train = pd.read_csv('data/train.csv', sep = '\t')
# df_test = pd.read_csv('data/test.csv', sep = '\t')

df_train.fillna('nothing', inplace = True)
df_test.fillna('nothing', inplace = True)

# train_lemtxt = df_train['lemm_text'].values
# train_lab = df_train['off_label'].values

# print(train_lemtxt)

offens_pol, non_offens_pol = build_offens_and_non_offens_polarity(train_lemtxt, train_lab)
bad_words = load_bad_words()

df_train['num_bad_words'] = df_train['text'].apply(lambda x: len([w for w in x.split(' ') if w in bad_words]))
df_test['num_bad_words'] = df_test['text'].apply(lambda x: len([w for w in x.split(' ') if w in bad_words]))

def get_offens_polarity(sent):
	t = sent.split(" ")
	acc_pol = 0
	for word in t:
		if word in offens_pol:
			acc_pol = offens_pol[word]
	return acc_pol


def get_non_offens_polarity(sent):
	t = sent.split(" ")
	acc_pol = 0
	for word in t:
		if word in non_offens_pol:
			acc_pol = non_offens_pol[word]
	return acc_pol

df_train['offen_pol'] = df_train['lemm_text'].apply(lambda x: get_offens_polarity(x))
df_test['offen_pol'] = df_test['lemm_text'].apply(lambda x: get_offens_polarity(x))

df_train['non_offen_pol'] = df_train['lemm_text'].apply(lambda x: get_non_offens_polarity(x))
df_test['non_offen_pol'] = df_test['lemm_text'].apply(lambda x: get_non_offens_polarity(x))

df_train['num_words'] = df_train['text'].apply(lambda x: len(x.split(" ")))
df_test['num_words'] = df_test['text'].apply(lambda x: len(x.split(" ")))

df_train['text_len'] = df_train['text'].apply(lambda x: len(x))
df_test['text_len'] = df_test['text'].apply(lambda x: len(x))

sid = SentimentIntensityAnalyzer()

df_train['vad_comp'] = df_train['text'].apply(lambda x: sid.polarity_scores(x)['compound'])
df_test['vad_comp'] = df_test['text'].apply(lambda x: sid.polarity_scores(x)['compound'])

df_train['vad_pos'] = df_train['text'].apply(lambda x: sid.polarity_scores(x)['pos'])
df_test['vad_pos'] = df_test['text'].apply(lambda x: sid.polarity_scores(x)['pos'])

df_train['vad_neg'] = df_train['text'].apply(lambda x: sid.polarity_scores(x)['neg'])
df_test['vad_neg'] = df_test['text'].apply(lambda x: sid.polarity_scores(x)['neg'])

df_train['vad_neu'] = df_train['text'].apply(lambda x: sid.polarity_scores(x)['neu'])
df_test['vad_neu'] = df_test['text'].apply(lambda x: sid.polarity_scores(x)['neu'])

df_train.to_csv('data/train.csv', index=False, sep='\t')
df_test.to_csv('data/test.csv', index=False, sep='\t')

X_train = df_train.loc[:,['num_bad_words', 'offen_pol', 'non_offen_pol', 'num_words', 'text_len', 'vad_comp', 'vad_pos', 'vad_neg', 'vad_neu']].values
y_train = df_train['off_label'].values

X_test = df_test.loc[:,['num_bad_words', 'offen_pol', 'non_offen_pol', 'num_words', 'text_len', 'vad_comp', 'vad_pos', 'vad_neg', 'vad_neu']].values
y_test = df_test['off_label'].values


# Random baseline
import random
rand_pred = []
for i in range(len(y_test)):
  rand_num = random.randint(1,101)
  if(rand_num<=50):
    rand_pred.append(0)
  else:
    rand_pred.append(1)

print("Random Baseline Score: "+f1_score(y_test, rand_pred, average='macro'))

fmacro_score = make_scorer(f1_score, average='macro')

# Random Forest Model
rf = RandomForestClassifier(random_state=42)
rf_grid = {'n_estimators':[10,50,75,100,150,200],
	'criterion':['gini','entropy'],
	'max_depth':[3,5,7],
	'min_samples_split':[10,20,50]
	}
rf = GridSearchCV(rf, rf_grid, cv=10, scoring=fmacro_score)

rf.fit(X_train,y_train)
print("Best CV score Random Forest:" + str(rf.best_score_))
print(rf.best_params_)

y_pred_rf = rf.predict(X_test)
rf_score = f1_score(y_test, y_pred_rf, avergae='macro')
print("Best Test Score Random Forest: "+str(rf_score))
dump(rf.best_estimator_,'model/rf_best_model_'+str(rf_score)+'.joblib')


# SVM Model
svm = SVC(random_state=42)
# reduce grid size for svm takes tooooo much time
svm_grid = {'C':[0.01, 0.1, 1, 10],
	'kernel': ['poly', 'rbf',],
	'gamma':['auto']
	}
svm = GridSearchCV(svm, svm_grid, cv=7, scoring=fmacro_score, n_jobs=20, verbose=2)

svm.fit(X_train,y_train)
print("Best CV score Random Forest:" + str(svm.best_score_))
print(svm.best_params_)

y_pred_svm = svm.predict(X_test)
svm_score = f1_score(y_test, y_pred_svm, avergae='macro')
print("Best Test Score SVM: "+str(svm_score))
dump(svm.best_estimator_,'model/svm_best_model_'+str(svm_score)+'.joblib')

# Logistic Model
logit = LogisticRegression()
logit_grid = {'penalty':['l1','l2'],
	'C' : [0.01, 0.1, 1, 10, 50]
	}
logit = GridSearchCV(logit, logit_grid, cv=10, scoring=fmacro_score, n_jobs=20, verbose=2)

logit.fit(X_train,y_train)
print("Best CV score Random Forest:" + str(logit.best_score_))
print(logit.best_params_)

y_pred_logit = logit.predict(X_test)
logit_score = f1_score(y_test, y_pred_logit, avergae='macro')
print("Best Test Score Logistic: "+str(logit_score))
dump(logit.best_estimator_,'model/logit_best_model_'+str(logit_score)+'.joblib')
