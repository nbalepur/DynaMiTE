import pickle
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import copy
import scipy
import math
import tqdm

# parameters to specify

data_folder = 'arxiv' # the folder which contains the dataset/results
topic_file_name = '../results/arxiv/nlp_cv_nn.txt' # the file which contains the topic evolution
num_topics = 5 # number of terms to consider when calculating NPMI 

def evaluate(curr_topics, texts):

    word2cnt = {}
    topk = len(curr_topics[0])

    for words in curr_topics:
        for word in words:
            word2cnt[word] = set()

    tot = 0.0
    # calculation smoothing regularizer
    epsilon = 1

    for idx, doc in enumerate(texts):
        tot += 1
        for word in word2cnt:
            if word in doc:
                word2cnt[word].add(idx)

    pmi = npmi = lcp = 0
    skip = True
    for words in curr_topics:
        
        for i in range(len(words)):
            for j in range(i):
                wi = words[i]
                wj = words[j]
                pi = (len(word2cnt[wi]) + epsilon) / tot
                pj = (len(word2cnt[wj]) + epsilon) / tot
                pij = (len(word2cnt[wi].intersection(word2cnt[wj])) + epsilon) / tot

                # PMI
                pmi += math.log(pij/(pi*pj))

                # NPMI
                npmi += -1 + math.log(pi*pj)/math.log(pij)

                # LCP
                lcp += math.log(pij/pj)

    word_set = set()
    for words in curr_topics:
        for word in words:
            word_set.add(word)

    num_topics = len(curr_topics)

    return pmi/len(topics)/(topk*(topk-1)), npmi/len(topics)/(topk*(topk-1)), lcp/len(topics)/(topk*(topk-1)), len(word_set)/((topk)*num_topics)

def parse_topic_file(topic_file_name):

    f = open(topic_file_name, 'r')
    all_lines = f.readlines()[2:]

    ret = []
    skip_num = all_lines.index("----------------------\n") + 1
    for i in range(0, len(all_lines), skip_num):
        topic_data = all_lines[i:i+skip_num]
        curr_topic_data = topic_data[2:-1]
        curr_topic_data = [t[t.index("["): t.index("]")+1] for t in curr_topic_data]
        curr_topic_data = [[item.replace("'", '') for item in t.strip('][').split(', ')] for t in curr_topic_data]
        ret.append(curr_topic_data)

    topic_names_ = [item[0] for item in ret[0]]

    topics = [[[x for x in val][1:num_topics+1] for val in data] for data in ret]

    f.close()
    return topics, topic_names_

data = []
topics, topic_names = parse_topic_file(topic_file_name)

df = pd.read_csv(f"../data/{data_folder}/data.csv", index_col = [0])
all_times = range(len(np.unique(df['time_discrete'])))

f = open(f"../data/{data_folder}/phrase_text.txt", "r")
all_docs = []
for line in f.readlines():
    all_docs.append(line.split())

bounds = np.cumsum(list(df.value_counts('time_discrete').sort_index()))
year_texts_phrase = []
for i in range(len(bounds)):
    if i == 0:
        year_texts_phrase.append(all_docs[:bounds[0]])
    else:
        year_texts_phrase.append(all_docs[bounds[i-1]:bounds[i]])

year_texts_phrase = dict(zip(list(all_times), year_texts_phrase))

total_pmi, total_npmi, total_lcp, total_div = 0.0, 0.0, 0.0, 0.0
for t in all_times:
    pmi, npmi, lcp, div = evaluate(topics[t], year_texts_phrase[t])
    total_pmi += pmi
    total_npmi += npmi
    total_lcp += lcp
    total_div += div

data.append([topic_file_name, topic_file_name, total_pmi / len(all_times), total_npmi / len(all_times), total_lcp / len(all_times), total_div / len(all_times)])

eval_df = pd.DataFrame(data, columns = ['folder', 'file', 'pmi', 'npmi', 'lcp', 'diversity'])

for file in np.unique(eval_df['file']):
    sub_df = eval_df[eval_df['file'] == file]
    print(sub_df)
    print('\n\n\n')