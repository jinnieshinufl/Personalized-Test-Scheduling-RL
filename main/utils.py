#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np 
import torch 

def data_import(train_dir='./', data='data.xlsx'):
    
    data = pd.read_excel(train_dir +'data.xlsx')
    data= data.fillna(0) # fill missing values with 0s
    
    data= data.drop(columns=['ID', 20])

    median = np.median(data.slope.tolist())
    data = data.astype(int)
    data['labels'] =((data['percentile'] > 25)
                     & (data['slope'] > median))*1

    labels = data.labels.to_numpy()
       
    data= data.drop(columns=['labels', 'slope', 'percentile'])
    data = data.to_numpy()   
   
    return data, labels 

def test_recommendation(questions_test):
    
    final_questions = []
    
    for sample in questions_test:
        final = []
        for i in range(len(sample[1])):
            temp = []
            for j in range(len(sample)):
                temp.append(sample[j][i].item())  
                temp = list(set(temp))
                temp.sort()
                
            final.append(temp)
        final_questions.append(final)
    recom = [item for sublist in final_questions for item in sublist]
    
    results = pd.DataFrame()
    results['ntest'] = results.original.apply(len)
    results = results[:len(recom)]
                
    results['recommend'] = recom            
    results['new_ntest'] = results.recommend.apply(len) 
    
    return results

def get_reward(rewards):
    R=0
    rewards_total = []
    for r in reversed(rewards):
        R = r + 0.99 * R
    rewards_total.insert(0, R)
    rewards_total = torch.cat(rewards_total, dim=1)
    return rewards_total

def test_score(batch, test):
    return batch[[i for i in range(len(test))], test], test

