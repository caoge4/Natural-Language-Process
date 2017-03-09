# -*- coding: utf-8 -*-
"""
Created on Sat Dec 03 09:34:49 2016

@author: Jinghui
"""
import os
import cPickle as cp
train_set=[]
train_label_set=[]
utt_set=[]
utt_label_set=[]
#test_set=[]
for root,dirs,files in os.walk("D:\Study\MyCode\data"):
    for file in files:
        if "pkl" in file: 
            file_dir = os.path.join(root,file)
            print file_dir
            train,val,test,dicts = cp.load(open(file_dir))
            train_set = train_set + train[0]
            train_label_set = train_label_set + train[2]
            #dicts = dicts
            w2idx, ne2idx, labels2idx = dicts['words2idx'], dicts['tables2idx'], dicts['labels2idx']
            idx2w = dict((v,k) for k,v in w2idx.items())
            idx2la = dict((v,k) for k,v in labels2idx.items())
            for utt_idx in train[0]:
                utt = map(lambda x: idx2w[x], utt_idx)
                utt_set.append(utt)
            for utt_label_idx in train[2]:
                utt_label = map(lambda x: idx2la[x], utt_label_idx)
                utt_label_set.append(utt_label)
max_x = max(map(lambda x: int(x),train_set))
            
