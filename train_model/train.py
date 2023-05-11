#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# main script for time CD 
# trainfile has lines of the form
# tok1,tok2,pmi

#from tkinter import E
import numpy as np
import util
from topic_ranker import TopicRanker
import pickle as pickle
import sys

# =================================== PARAMETER SPECIFICATION ===================================

T = range(2012, 2023) # list of time spans (in order)

seeds = [['natural_language_processing'], ['vision'], ['neural_network']] # seed words
should_update_seeds = True # should we update seeds?

# save output
out_file_name = "test"

# location of files
trainhead = 'data/arxiv' # location of training data
savehead = 'results/arxiv'

# iterations
num_topics = 5 # number of topics to mine
update_every = 1 # after how many iterations to update new topics
ITERS = num_topics * update_every

# parameters
lam = 100 # local contexts
tau = 50  # temporal contexts
kappa = 50 # discriminative loss
gam = 50 # forcing regularizer

beta = 0.05 # popularity for BM25

# =================================== END OF PARAMETER SPECIFICATION ===================================

emph = 1 # emphasize the nonzero
r = 50  # rank

param_str = f"lambda: {lam} | gamma: {gam} | tau: {tau} | kappa: {kappa} | num topics: {num_topics} | update every: {update_every} | beta: {beta}"
orig_seeds = seeds.copy() # original seed words
def print_params(r,lam,tau,gam,emph,ITERS,kappa):
    
    print('rank = {}'.format(r))
    print('frob  regularizer = {}'.format(lam))
    print('time  regularizer = {}'.format(tau))
    print('symmetry regularizer = {}'.format(gam))
    print('emphasize param   = {}'.format(emph))
    print('total iterations = {}'.format(ITERS))
    print(f'discriminative regularizer = {kappa}')

    
import pandas as pd
import copy

if __name__=='__main__':

    # get the word2id / id2word mappings
    word_to_id = dict()
    id_to_word= dict()
    df = pd.read_csv(f"{trainhead}/wordIDHash.csv", header = None)
    for key, row in df.iterrows():
        word_id, word, _ = list(row)
        id_to_word[word_id] = word
        word_to_id[word] = word_id

    with open(f"{trainhead}/temporal_tfidf.pkl", "rb") as handle:
        word_time_tfidf = pickle.load(handle)

    with open(f"{trainhead}/vocab.pkl", "rb") as handle:
        vocabs = pickle.load(handle)

    ranker = TopicRanker(word_to_id, id_to_word, word_time_tfidf, vocabs, T, trainhead, param_str)

    # number of words and batch size
    nw = len(word_to_id) 
    b = nw

    seed_ids = []
    for category in seeds:
        curr_ids = [word_to_id[word] for word in category]
        seed_ids.append(curr_ids)
    orig_wids = seed_ids.copy()

    seeds_t = [copy.deepcopy(seed_ids) for _ in T]

    foo = sys.argv
    for i in range(1,len(foo)):
        if foo[i]=='-r':    r = int(float(foo[i+1]))        
        if foo[i]=='-iters': ITERS = int(float(foo[i+1]))            
        if foo[i]=='-lam':    lam = float(foo[i+1])
        if foo[i]=='-tau':    tau = float(foo[i+1])
        if foo[i]=='-gam':    gam = float(foo[i+1])
        if foo[i]=='-b':    b = int(float(foo[i+1]))
        if foo[i]=='-emph': emph = float(foo[i+1])
        if foo[i]=='-check': erchk=foo[i+1]
    
    print('starting training with following parameters')
    print_params(r,lam,tau,gam,emph,ITERS,kappa)
    print('there are a total of {} words, and {} time points'.format(nw,T))
    
    print('X*X*X*X*X*X*X*X*X')
    print('initializing')

    if b < nw:
        b_ind = util.getbatches(nw,b)
    else:
        b_ind = [range(nw)]

    Ulist,Vlist = util.import_static_init(T, trainhead)

    all_pmis = [util.getmat(f"{trainhead}/wordPairPMI_{real_time}.csv", nw, False).todense() for t, real_time in enumerate(T)]

    # sequential updates
    for iteration in range(ITERS):  

        print(f"Iteration {iteration}")

        loss = 0
        # shuffle times
        if iteration == 0: times = T
        else: times = np.random.permutation(T)

        for t, real_time in enumerate(times):   # select a time
            print('iteration %d, time %d' % (iteration, t))

            pmi = all_pmis[t]

            if kappa > 0:

                # dissimilar seeds
                for c1_idx in range(len(seeds_t[t])):
                    for c2_idx in range(c1_idx + 1, len(seeds_t[t])):
                        c1, c2 = seeds_t[c1_idx], seeds_t[c2_idx]
                        for seed1 in c1:
                            for seed2 in c2:
                                pmi[seed1, seed2] = 0
                                pmi[seed2, seed1] = 0

                # similar seeds
                for category in seeds_t[t]:
                    for s1_idx in range(len(category)):
                        for s2_idx in range(s1_idx + 1, len(category)):
                            seed1, seed2 = category[s1_idx], category[s2_idx]
                            pmi[seed1, seed2] = 1
                            pmi[seed2, seed1] = 1

            for j in range(len(b_ind)): # select a mini batch
                ind = b_ind[j]
                
                if t==0:
                    vp = np.zeros((len(ind),r))
                    up = np.zeros((len(ind),r))
                    iflag = True
                else:
                    vp = Vlist[t-1][ind,:]
                    up = Ulist[t-1][ind,:]
                    iflag = False

                if t==len(T)-1:
                    vn = np.zeros((len(ind),r))
                    un = np.zeros((len(ind),r))
                    iflag = True
                else:
                    vn = Vlist[t+1][ind,:]
                    un = Ulist[t+1][ind,:]
                    iflag = False

                # update step for full computation
                # Vlist[t][ind,:] = util.update(Ulist[t],emph*pmi,vp,vn,lam,tau,gam,ind,iflag,seeds_t[t],kappa)
                # Vlist[t][ind,:] = util.update(Vlist[t],emph*pmi,up,un,lam,tau,gam,ind,iflag,seeds_t[t],kappa)

                # update step for fast computation
                Vlist[t][ind,:] = util.update_fast(Ulist[t],emph*pmi,vp,vn,lam,tau,gam,ind,iflag)
                Ulist[t][ind,:] = util.update_fast(Vlist[t],emph*pmi,up,un,lam,tau,gam,ind,iflag)

        if should_update_seeds and (iteration % update_every == 0 and (iteration != 0 or update_every == 1)):
            seeds_t = ranker.update_seeds_bm25_tfidf(seeds_t, Ulist, T, beta)
            ranker.print_seeds(seeds_t)

    # save the embeddings and output file
    pickle.dump(Ulist, open(f"{savehead}/{out_file_name}.pkl", "wb"), pickle.HIGHEST_PROTOCOL)
    if should_update_seeds:
        print("Final Results\n==============")
        ranker.print_seeds(seeds_t, output_file=f"{savehead}/{out_file_name}.txt")