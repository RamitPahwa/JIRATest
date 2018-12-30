# -*- coding: utf-8 -*-
"""SemEval19.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fNqfpXSKPWfaCPHdjiVi3prbtPFE6RKu
"""

import numpy as np
import pandas as pd

!ls

! pip install fastai==0.7.0
! pip install torchtext==0.2.3
! pip install sklearn

from fastai.text import *
import html

DATA_PATH=Path('data/')

DATA_PATH.mkdir(exist_ok=True)

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

# !pip install ekphrasis
# !pip install nltk

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.classes.spellcorrect import SpellCorrector

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
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

def clean_data(corpus):
    new_corp = []
    # spell correct
    for i in range(len(corpus)):
        t = text_processor.pre_process_doc(corpus[i])
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

print(lemmatized_text)

trn_texts,val_texts = sklearn.model_selection.train_test_split(
    lemmatized_text, test_size=0.1)

col_names = ['labels','text']
np.random.seed(42)
trn_idx = np.random.permutation(len(trn_texts))
val_idx = np.random.permutation(len(val_texts))

trn_texts = np.asarray(trn_texts)
val_texts = np.asarray(val_texts)

trn_texts = trn_texts[trn_idx]
val_texts = val_texts[val_idx]

trn_labels = label_a[trn_idx]
val_labels = label_a[val_idx]

df_trn = pd.DataFrame({'text':trn_texts, 'labels':trn_labels}, columns=col_names)
df_val = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=col_names)

df_trn.to_csv(DATA_PATH/'train.csv', header=False, index=False)
df_val.to_csv(DATA_PATH/'test.csv', header=False, index=False)

trn_texts,val_texts = sklearn.model_selection.train_test_split(
    np.concatenate([trn_texts,val_texts]), test_size=0.1)

df_trn = pd.DataFrame({'text':trn_texts, 'labels':[0]*len(trn_texts)}, columns=col_names)
df_val = pd.DataFrame({'text':val_texts, 'labels':[0]*len(val_texts)}, columns=col_names)

df_trn.to_csv(DATA_PATH/'train_lm.csv', header=False, index=False)
df_val.to_csv(DATA_PATH/'test_lm.csv', header=False, index=False)

re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

chunksize=24000

def get_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)

def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels

df_trn = pd.read_csv(DATA_PATH/'train_lm.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(DATA_PATH/'test_lm.csv', header=None, chunksize=chunksize)

tok_trn, trn_labels = get_all(df_trn, 1)
tok_val, val_labels = get_all(df_val, 1)

freq = Counter(p for o in tok_trn for p in o)
freq.most_common(25)

max_vocab = 60000
min_freq = 2

itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
itos.insert(0, '_pad_')
itos.insert(0, '_unk_')

stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
len(itos)


trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
val_lm = np.array([[stoi[o] for o in p] for p in tok_val])

np.save(DATA_PATH/'trn_ids_lm.npy', trn_lm)
np.save(DATA_PATH/'val_ids_lm.npy', val_lm)
pickle.dump(itos, open(DATA_PATH/'itos_lm.pkl', 'wb'))

trn_lm = np.load(DATA_PATH/'trn_ids_lm.npy')
val_lm = np.load(DATA_PATH/'val_ids_lm.npy')
itos = pickle.load(open(DATA_PATH/'itos_lm.pkl', 'rb'))

vs=len(itos)
vs,len(trn_lm)

# ! wget -nH -r -np -P {DATA_PATH} http://files.fast.ai/models/wt103/

em_sz,nh,nl = 400,1150,3
PRE_PATH = DATA_PATH/'models'/'wt103'
PRE_LM_PATH = PRE_PATH/'fwd_wt103.h5'

wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)
enc_wgts = to_np(wgts['0.encoder.weight'])
row_m = enc_wgts.mean(0)  

itos2 = pickle.load((PRE_PATH/'itos_wt103.pkl').open('rb'))
stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})

new_w = np.zeros((vs, em_sz), dtype=np.float32)

for i,w in enumerate(itos):
    r = stoi2[w]
    new_w[i] = enc_wgts[r] if r>=0 else row_m

wgts['0.encoder.weight'] = T(new_w)
wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
wgts['1.decoder.weight'] = T(np.copy(new_w))

wd=1e-7
bptt=70
bs=52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(DATA_PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7

learner= md.get_model(opt_fn, em_sz, nh, nl, 
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy,f1]
learner.freeze_to(-1)

learner.model.load_state_dict(wgts)

lr=1e-3
lrs = lr

learner.fit(lrs/2, 1, wds=wd, use_clr=(32,2), cycle_len=1)

learner.save('lm_last_ft')
learner.load('lm_last_ft')
learner.unfreeze()
learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)
learner.sched.plot()
learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=15)
learner.save('lm1')
learner.save_encoder('lm1_enc')
learner.sched.plot_loss()

df_trn = pd.read_csv(DATA_PATH/'train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(DATA_PATH/'test.csv', header=None, chunksize=chunksize)

tok_trn, trn_labels = get_all(df_trn, 1)
tok_val, val_labels = get_all(df_val, 1)

(DATA_PATH/'tmp').mkdir(exist_ok=True)

np.save(DATA_PATH/'tmp'/'tok_trn.npy', tok_trn)
np.save(DATA_PATH/'tmp'/'tok_val.npy', tok_val)

np.save(DATA_PATH/'tmp'/'trn_labels.npy', trn_labels)
np.save(DATA_PATH/'tmp'/'val_labels.npy', val_labels)

tok_trn = np.load(DATA_PATH/'tmp'/'tok_trn.npy')
tok_val = np.load(DATA_PATH/'tmp'/'tok_val.npy')

itos = pickle.load((DATA_PATH/'itos_lm.pkl').open('rb'))
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
len(itos)

trn_clas = np.array([[stoi[o] for o in p] for p in tok_trn])
val_clas = np.array([[stoi[o] for o in p] for p in tok_val])

np.save(DATA_PATH/'tmp'/'trn_ids.npy', trn_clas)
np.save(DATA_PATH/'tmp'/'val_ids.npy', val_clas)

trn_clas = np.load(DATA_PATH/'tmp'/'trn_ids.npy')
val_clas = np.load(DATA_PATH/'tmp'/'val_ids.npy')

trn_labels = np.squeeze(np.load(DATA_PATH/'tmp'/'trn_labels.npy'))
val_labels = np.squeeze(np.load(DATA_PATH/'tmp'/'val_labels.npy'))

bptt,em_sz,nh,nl = 70,400,1150,3
vs = len(itos)
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
bs = 48

min_lbl = trn_labels.min()
trn_labels -= min_lbl
val_labels -= min_lbl
c=int(trn_labels.max())+1

"""In the classifier, unlike LM, we need to read a movie review at a time and learn to predict the it's sentiment as pos/neg. We do not deal with equal bptt size batches, so we have to pad the sequences to the same length in each batch. To create batches of similar sized movie reviews, we use a sortish sampler method invented by [@Smerity](https://twitter.com/Smerity) and [@jekbradbury](https://twitter.com/jekbradbury)

The sortishSampler cuts down the overall number of padding tokens the classifier ends up seeing.
"""

trn_ds = TextDataset(trn_clas, trn_labels)
val_ds = TextDataset(val_clas, val_labels)
trn_samp = SortishSampler(trn_clas, key=lambda x: len(trn_clas[x]), bs=bs//2)
val_samp = SortSampler(val_clas, key=lambda x: len(val_clas[x]))
trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
md = ModelData(DATA_PATH, trn_dl, val_dl)

# part 1
dps = np.array([0.4, 0.5, 0.05, 0.3, 0.1])

dps = np.array([0.4,0.5,0.05,0.3,0.4])*0.5

m = get_rnn_classifer(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
          layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
          dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

opt_fn = partial(optim.Adam, betas=(0.7, 0.99))

learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learn.clip=.25
learn.metrics = [accuracy]

lr=3e-3
lrm = 2.6
lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])

lrs=np.array([1e-4,1e-4,1e-4,1e-3,1e-2])

wd = 1e-7
wd = 0
learn.load_encoder('lm1_enc')

learn.freeze_to(-1)

learn.lr_find(lrs/1000)
learn.sched.plot()

learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))

learn.save('clas_0')

learn.load('clas_0')

learn.freeze_to(-2)

learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))

learn.save('clas_1')

learn.load('clas_1')

learn.unfreeze()

learn.fit(lrs, 1, wds=wd, cycle_len=14, use_clr=(32,10))

learn.sched.plot_loss()

learn.save('clas_2')

learn.sched.plot_loss()

learn.load('clas_1')
learn.unfreeze()

learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(32,10))

