import pickle
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import copy
import scipy
import math

data_folder = 'arxiv' # data folder
all_times = range(2012, 2023) # time spans
filename = 'nlp_cv_nn' # name of the embedding/text file
results = 'results' # results folder

def parse_topic_file():

    f = open(f"../{results}/{data_folder}/{filename}.txt", 'r')
    all_lines = f.readlines()[2:]

    ret = []
    skip_num = all_lines.index("----------------------\n") + 1
    for i in range(0, len(all_lines), skip_num):
        topic_data = all_lines[i:i+skip_num]
        curr_topic_data = topic_data[2:-1]
        curr_topic_data = [t[t.index("["): t.index("]")+1] for t in curr_topic_data]
        curr_topic_data = [[item.replace("'", '') for item in t.strip('][').split(', ')] for t in curr_topic_data]
        ret.append(curr_topic_data)

    topic_names = [item[0] for item in ret[0]]
    topics = [[[x for x in val[1:6]] for val in data] for data in ret]

    f.close()
    return topics, topic_names

topics, topic_names = parse_topic_file()

with open(f'../{results}/{data_folder}/{filename}.pkl', 'rb') as handle:
    embeds = pickle.load(handle)

word_to_id = dict()
id_to_word= dict()
df = pd.read_csv(f"../data/{data_folder}/wordIDHash.csv", header = None)
for key, row in df.iterrows():
    word_id, word, _ = list(row)
    id_to_word[word_id] = word
    word_to_id[word] = word_id

for i, topic in enumerate(topic_names):

    vals = []
    for t in range(len(embeds) - 1):
        cos_sim_val = cosine_similarity(embeds[t][word_to_id[topic]].reshape(1, -1), embeds[t+1][word_to_id[topic]].reshape(1, -1))[0][0]
        vals.append(cos_sim_val)

    import numpy as np
    vals = np.array(vals)
    check_year_shift = np.argmin(vals) + 1

    word_sims = []
    for term in topics[check_year_shift][i][:10]:
        word_sims.append(cosine_similarity(embeds[check_year_shift - 1][word_to_id[term]].reshape(1, -1), embeds[check_year_shift][word_to_id[term]].reshape(1, -1))[0][0])
    word_sims = np.array(word_sims)

    check_year_shift1 = np.argmax(vals) + 1
    word_sims1 = []
    for term in topics[check_year_shift1][i][:10]:
        word_sims1.append(cosine_similarity(embeds[check_year_shift - 1][word_to_id[term]].reshape(1, -1), embeds[check_year_shift][word_to_id[term]].reshape(1, -1))[0][0])
    word_sims1 = np.array(word_sims1)

    print(f"Category name {topic} shifted the most from {list(all_times)[check_year_shift-1]}-{list(all_times)[check_year_shift]}")
    print(f"Top 3 most shifting words:")
    print([topics[check_year_shift][i][idx] for idx in np.argsort(word_sims)][:3])

    print('\n')