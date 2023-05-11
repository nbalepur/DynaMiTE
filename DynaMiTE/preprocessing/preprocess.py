# parameters to modify

data_folder_name = 'arxiv' # folder name which contains your file 'data.csv'. Must have columns 'text' and 'time_discrete'
all_times = range(2012, 2023) # list of time spans (in order)
vocab_freq = 1/5000 # frequency of documents that must contain a word in order for the word to be included in the corpus
ppmi_window_size = 7 # PPMI window size

tfidf_window_size = 5 # Temporal TF-IDF window size

stop_after_autophrase = False # should stop the processing after autophrase?
use_autophrase = True # should use autophrase or raw text for vocabulary?

# ------------------------------ Preparing Data for AutoPhrase ------------------------------

CRED = '\33[44m'
CEND = '\033[0m'
print()
print(CRED + "Step 1: Preparing data for AutoPhrase" + CEND)
print()

import pandas as pd
import numpy as np
import tqdm
import nltk
data_pref = f"../data/{data_folder_name}"

df = pd.read_csv(f"{data_pref}/data.csv")[['text', 'time_discrete']]

if all_times == None:
    all_times = list(np.unique(df['time_discrete']))

window_size = ppmi_window_size
auto_phrase_file = open(f"{data_pref}/text.txt", "w+")
cumulative_count = [0]

for time in tqdm.tqdm(all_times):
    df_subset = df[df['time_discrete'] == time]
    cumulative_count.append(cumulative_count[-1] + len(df_subset))
    for key, row in df_subset.iterrows():
        text, time = row
        text = text.replace("\n", " ")
        #text = ' '.join(nltk.word_tokenize(text)[:750])
        auto_phrase_file.write(f"{text}\n")

cumulative_count.pop(0)
auto_phrase_file.close()

# ------------------------------ Running AutoPhrase ------------------------------
print()
print(CRED + "Step 2: Running AutoPhrase" + CEND)
print()

if use_autophrase:

    def execute(cmd):
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        for stdout_line in iter(popen.stdout.readline, ""):
            yield stdout_line 
        popen.stdout.close()
        return_code = popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)

    import subprocess
    for path in execute(['./auto_phrase.sh', '-f', data_folder_name]):
        print(path, end="")

    print('Done running autophrase\n\n')

    if stop_after_autophrase:
        exit(0)

# ------------------------------ Preparing Embedding Data ------------------------------

from collections import Counter
from itertools import combinations
from math import log
import numpy as np
import pandas as pd
from pprint import pformat
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, norm
from string import punctuation
import pickle
import nltk
import numpy as np
from nltk.corpus import stopwords
import tqdm

print(CRED + "Step 3: Preparing Embedding Data" + CEND)
print()

fname = f"{data_pref}/phrase_text.txt" if use_autophrase else f"{data_pref}/text.txt"
f = open(fname, "r")
all_docs = []
for line in f.readlines():
    all_docs.append(line.split())

stop_words = stopwords.words('english')

year_texts_phrase = []
for i in range(len(cumulative_count)):
    if i == 0:
        year_texts_phrase.append(all_docs[:cumulative_count[0]])
    else:
        year_texts_phrase.append(all_docs[cumulative_count[i-1]:cumulative_count[i]])

year_texts_phrase = dict(zip(list(all_times), year_texts_phrase))

print("Processing vocab in corpus")

cx_base = Counter()
cxy_base = Counter()
cx_doc = Counter()
for text in tqdm.tqdm(all_docs):
    for x in text:
        cx_base[x] += 1
    for i, x in enumerate(text):
        window = text[max(i-(window_size//2), 0):min(i+(window_size//2 + 1),len(text))]
        for y in window:
            if x == y:
                continue
            cxy_base[(min(x, y), max(x, y))] += 1

#print('%d tokens before' % len(cx))
num_rem = 0
min_count = vocab_freq * len(df)
for x in list(cx_base.keys()):
    if type(x) != type('') or x in stop_words or cx_base[x] <= min_count or x.replace("-", "").replace("/", "").isnumeric() or x in {'null', 'nan'}:
        num_rem += 1
        del cx_base[x]
#print('%d tokens after' % len(cx))
#print('Most common:', cx.most_common()[:25])

for x, y in list(cxy_base.keys()):
    if x not in cx_base or y not in cx_base:
        del cxy_base[(x, y)]

x2i_base, i2x_base = {}, {}
for i, x in enumerate(cx_base.keys()):
    x2i_base[x] = i
    i2x_base[i] = x
    
hash_id = []
for key, val in i2x_base.items():
    hash_id.append([key, val, cx_base[val]])
pd.DataFrame(hash_id).to_csv(f"{data_pref}/wordIDHash.csv", index=False, header=None)

def create_pmi(text_corpus):

    cx = Counter()
    cxy = Counter()
    for text in tqdm.tqdm(text_corpus):
        for x in text:
            cx[x] += 1
        for i, x in enumerate(text):
            window = text[max(i-(window_size//2), 0):min(i+(window_size//2 + 1),len(text))]
            for y in window:
                if x == y:
                    continue
                cxy[(min(x, y), max(x, y))] += 1
       
    for x in list(cx.keys()):        
        if x not in x2i_base:
            del cx[x]
            
    x2i, i2x = {}, {}
    for i, x in enumerate(cx.keys()):
        x2i[x] = i
        i2x[i] = x

    for x, y in list(cxy.keys()):
        if x not in cx or y not in cx:
            del cxy[(x, y)]

    sx = sum(cx.values())
    sxy = sum(cxy.values())

    pmi_samples = Counter()
    data, rows, cols = [], [], []
    for (x, y), n in cxy.items():
        rows.append(x2i[x])
        cols.append(x2i[y])

        # normalized PMI
        num = log((cx[x] / sx) * (cx[y] / sx))
        den = log(n / sxy)
        data.append((num / den) - 1)
        
        pmi_samples[(x, y)] = max(data[-1], 0)
    PMI = csc_matrix((data, (rows, cols)))
    PPMI = np.where(PMI.toarray() >= 0, PMI.toarray(), 0)
    
    U, _, _ = svds(PPMI, k=50)
    
    U_ret = np.zeros((len(x2i_base), 50))
    for i in range(len(U)):
        U_ret[x2i_base[i2x[i]], :] = U[i, :]
        
    return U_ret, pmi_samples, cx, cxy, set(x2i.keys())

static_emb_local = []
all_randoms = []
counters = []
vocabs = []

print("\n\nCalculating PPMI for each time step")

for year in all_times:
    emb, pmi_samples, cx, cxy, vocab = create_pmi(year_texts_phrase[year])
    
    pmi_data = []
    for row in pmi_samples:
        pmi_data.append([x2i_base[row[0]], x2i_base[row[1]], pmi_samples[row]])
    pd.DataFrame(pmi_data).to_csv(f"{data_pref}/wordPairPMI_{year}.csv", index=False)
    
    static_emb_local.append(emb)
    counters.append((cx, cxy))
    vocabs.append(vocab)
    
with open(f"{data_pref}/static_emb_local.pkl", "wb") as handle:
    pickle.dump(static_emb_local, handle)

with open(f"{data_pref}/vocab.pkl", "wb") as handle:
    pickle.dump(vocabs, handle)

print("\n\nCalculating overall PPMI")

import copy
U_ret, _, _, _, _ = create_pmi(all_docs)
with open(f"{data_pref}/static_emb_global.pkl", "wb") as handle:
    static_emb_global = [copy.deepcopy(U_ret) for _ in all_times]
    pickle.dump(static_emb_global, handle)

from itertools import product
import tqdm

from itertools import product
import tqdm

def decay_function(time_diff, window_size):
    return np.exp(-(5 / window_size) * time_diff)

num_words_dist = [sum([len(doc) for doc in year_texts_phrase[t]]) for t in all_times]

def tf_itf(w, t, counters, discr_window_size = 5, cont_window_size = 20000):
    
    if w not in counters[t][0]:
        return 0, 0, 0
    
    tf = 0.0
        
    min_bound, max_bound = t - discr_window_size // 2, t + discr_window_size // 2 + 1
    
    if max_bound >= len(counters):
        min_bound -= (max_bound - len(counters))
        max_bound = len(counters)
    
    if min_bound < 0:
        max_bound -= min_bound
        min_bound = 0
        
    window_size = max_bound - min_bound
    
    tf += (1.0 * counters[t][0][w]) / (num_words_dist[t]) 
        
    itf = (window_size) / (1.0 * sum([int(w in ctr[0]) for ctr in counters[min_bound:max_bound]]))
    
    return (tf) * np.log(itf), tf, itf

def tf_itf_flat_grow(w, t, counters, back_dir, fwd_dir):
    
    if w not in counters[t][0]:
        return 0, 0, 0
    
    tf = 0.0
        
    min_bound, max_bound = t - back_dir - 1, t + fwd_dir

    if max_bound >= len(counters):
        min_bound -= (max_bound - len(counters))
        max_bound = len(counters)
    
    if min_bound < 0:
        max_bound -= min_bound
        min_bound = 0
    
    window_size = max_bound - min_bound
    
    tf = 0.0
    for new_t in range(min_bound, max_bound):
        if new_t >= t:
            tf += counters[new_t][0][w]
    
    itf = ((t - min_bound + 1)) / (1.0 * sum([int(w in ctr[0]) for ctr in counters[min_bound:t+1]]))
    
    return (tf / (max_bound - t)) * np.log(itf), tf, itf

def tf_itf_flat_decay(w, t, counters, back_dir, fwd_dir):
    
    if w not in counters[t][0]:
        return 0, 0, 0
        
    min_bound, max_bound = t - back_dir - 1, t + fwd_dir

    if max_bound >= len(counters):
        min_bound -= (max_bound - len(counters))
        max_bound = len(counters)
    
    if min_bound < 0:
        max_bound -= min_bound
        min_bound = 0
    
    window_size = max_bound - min_bound
    
    tf = 0.0
    for new_t in range(min_bound, max_bound):
        if new_t <= t:
            tf += counters[new_t][0][w]
    
    tf = max(tf, 0)
    
    if w not in counters[t][0]:
        return 0.0, 0.0, 0.0
    
    itf = ((max_bound - t)) / (1.0 * sum([int(w in ctr[0]) for ctr in counters[t:max_bound]]))
    
    return (tf / (t - min_bound + 1)) * np.log(itf), tf, itf

print("\n\nCalculating temporal TF-IDF scores")

word_time_tfidf_burst = dict()
for w, i in tqdm.tqdm(product(list(x2i_base.keys()), list(all_times)), total=len(x2i_base) * len(all_times)):
    word_time_tfidf_burst[(i, w)] = tf_itf(w, list(all_times).index(i), counters, discr_window_size = tfidf_window_size, cont_window_size = 20000)[0]
    
word_time_tfidf_grow = dict()
for w, i in tqdm.tqdm(product(list(x2i_base.keys()), list(all_times)), total=len(x2i_base) * len(all_times)):
    word_time_tfidf_grow[(i, w)] = tf_itf_flat_grow(w, list(all_times).index(i), counters, back_dir = tfidf_window_size // 2, fwd_dir = tfidf_window_size // 2)[0]

with open(f"{data_pref}/temporal_tfidf.pkl", "wb") as handle:
    pickle.dump({'burst': word_time_tfidf_burst}, handle)

print()
print(CRED + "Data has been fully processed!" + CEND)