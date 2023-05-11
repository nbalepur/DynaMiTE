import numpy as np
from nltk.stem import WordNetLemmatizer
import copy
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import pandas as pd

class TopicRanker():

    def __init__(self, word_to_id, id_to_word, word_time_tfidf, vocab, all_times, trainhead, param_str):

        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.word_time_tfidf = word_time_tfidf
        self.vocab = vocab
        self.param_str = param_str
        self.lemmatizer = WordNetLemmatizer()

        if word_time_tfidf != None:
            self.tfidf_ranks = [scipy.stats.rankdata(-1 * np.array([self.word_time_tfidf['burst'][t, self.id_to_word[wid]] for wid in range(len(self.id_to_word))]), method='max') for t in all_times]

        # initialize year_texts_phrase
        f = open(f"{trainhead}/phrase_text.txt", "r")
        all_docs = []
        for line in f.readlines():
            all_docs.append(line.split())

        df = pd.read_csv(f"{trainhead}/data.csv", index_col = [0])

        bounds = np.cumsum(list(df.value_counts('time_discrete').sort_index()))
        year_texts_phrase = []
        for i in range(len(bounds)):
            if i == 0:
                year_texts_phrase.append(all_docs[:bounds[0]])
            else:
                year_texts_phrase.append(all_docs[bounds[i-1]:bounds[i]])

        year_texts_phrase = dict(zip(list(all_times), year_texts_phrase))
        self.all_times = all_times
        self.year_texts_phrase = year_texts_phrase

    def isfloat(self, num):
        try:
            float(num)
            return True
        except ValueError:
            return False

    # prints seeds (and saves them if specified)
    def print_seeds(self, curr_seeds, output_file = None):

        out_str = f"{self.param_str}\n\n"
        for t, time_seeds in enumerate(curr_seeds):
            print(f"Time t={t}\n")
            out_str += f"Time t={t}\n\n"
            for category in time_seeds:
                print(f"{self.id_to_word[category[0]]}: {[self.id_to_word[wid] for wid in category]}")
                out_str += f"{self.id_to_word[category[0]]}: {[self.id_to_word[wid] for wid in category]}\n"
            print("----------------------")
            out_str += "----------------------\n"

        if output_file != None:
            f = open(output_file, "w+")
            f.write(out_str)
            f.close()

    def topic_exists_lemm(self, idx, curr_wids):
        lemm_word = self.lemmatizer.lemmatize(self.id_to_word[idx])
        curr_words = set([self.lemmatizer.lemmatize(self.id_to_word[wid]) for wid in curr_wids])
        return lemm_word in curr_words


    def construct_docs(self, categories, times):
    
        doc_mat = [[[] for _ in categories[0]] for _ in times]
        
        for t, time in enumerate(times):
            for c, category in enumerate(categories[t]):
                for doc in self.year_texts_phrase[time]:
                    
                    found = False
                    for cat_seed in category:
                        if cat_seed in doc:
                            found = True
                            break
                    
                    if found:
                        doc_mat[t][c].extend(doc)
                            
        return doc_mat

    def get_cat_bm25(self, doc_mat, time):
        
        bm25 = BM25Okapi(doc_mat[time])
        
        scores = []
        freqs = []

        for i in range(len(self.word_to_id)):
            curr_scores = bm25.get_scores([self.id_to_word[i]])
            scores.append(np.exp(curr_scores) / np.sum(np.exp(curr_scores)))
            freqs.append([bm25.doc_freqs[t].get(self.id_to_word[i], 0) for t in range(len(doc_mat[time]))])
            
        scores = np.array(scores).T
        if scores.shape[0] == 2:
            scores[[0, 1]] = scores[[1, 0]]
            
        return np.array(freqs).T, scores

    def get_time_bm25(self, doc_mat, category, all_times):
        
        bm25 = BM25Okapi([doc_mat[t][category] for t in range(len(all_times))])
        
        scores = []
        freqs = []

        for i in range(len(self.word_to_id)):
            curr_scores = bm25.get_scores([self.id_to_word[i]])
            scores.append(np.exp(curr_scores) / np.sum(np.exp(curr_scores)))
            freqs.append([bm25.doc_freqs[t].get(self.id_to_word[i], 0) for t in range(len(all_times))])
            
        return np.array(freqs).T, np.array(scores).T

    def update_seeds_bm25_tfidf(self, seeds_t, U_t, all_times, beta):

        seeds_t = seeds_t.copy()
        seeds_t_words = [[[self.id_to_word[x] for x in y] for y in z] for z in seeds_t]

        doc_mat = self.construct_docs(seeds_t_words, all_times)
        cat_bm25_data = [self.get_cat_bm25(doc_mat, i) for i in range(len(all_times))]
        
        for t, U_in in enumerate(U_t):
            for i, curr_wids in enumerate(seeds_t[t]):

                emb_avg = np.mean(U_in[curr_wids, :], axis = 0)

                other_seeds = set()
                for j, other_wids in enumerate(seeds_t[t]):
                    if j == i:
                        continue
                    for wid in other_wids:
                        other_seeds.add(wid)

                sim_scores = cosine_similarity(U_in, emb_avg.reshape(1, -1)).flatten()
                
                cat_popularity = np.log(cat_bm25_data[t][0][i] + 1)
                cat_distinct = cat_bm25_data[t][1][i]
                cat_scores = (cat_popularity**beta) * (cat_distinct**(1 - beta))

                tfidf_rank = self.tfidf_ranks[t]
                sim_rank = scipy.stats.rankdata(-1 * sim_scores, method='max')
                cat_rank = scipy.stats.rankdata(-1 * cat_scores, method='max')
                
                best_idx = np.argsort(sim_rank + tfidf_rank + cat_rank)

                for idx in best_idx:
                    if not idx in curr_wids and self.id_to_word[idx] in self.vocab[t] and not idx in other_seeds and self.id_to_word[idx] not in {'-', 'well-'}:
                        seeds_t[t][i].append(idx)
                        break

        return seeds_t