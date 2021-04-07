#!/usr/bin/env python
# coding: utf-8

##########################################
# modified from the original code by Nurakhmetov (2019)
'''
references:
Nurakhmetov, D. (2019). Reinforcement Learning Applied to Adaptive Classification Testing. In Theoretical and Practical Advances in Computer-based Educational Measurement (pp. 325-336). Springer, Cham.    
'''
import numpy as np
import tensorflow as tf 
from tf.keras.layers import Embedding, GRU,concatenate, Dense 
from tf.keras import activations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


def policy_model(question, point, hidden, data_batch):
    
    score_range = 1401
    test_size =18 
    batch_size = 10
    
    if (len(question) < 1) & (len(point) <1):
        question = np.random.randint(1, test_size+1, size=(batch_size))
        point = np.random.randint(0, 3, size=(batch_size))
    else:
        question = question 
        point = point 
        
    question = np.reshape(question, (len(question),1))
    point = np.reshape(point, (len(point),1))
    input_emb = Embedding(output_dim=test_size, input_dim=test_size+1, input_length=batch_size, name='item_emb')(question)
    score_emb = Embedding(output_dim=test_size, input_dim=  score_range+1, input_length=batch_size, name='score_emb')(point)

    score_emb = activations.relu(score_emb)
    input_emb = activations.relu(input_emb)
    
    embedded = concatenate([input_emb, score_emb], 2 )
    gru = GRU(test_size)(embedded)
    logits = Dense( test_size, activation='linear')(hidden)
    probability = activations.softmax(logits)
    
    test_probability = tf.random.categorical(probability, 1).numpy()
    score_probability = data_batch[[i for i in range(len(test_probability.flatten()))],
                                                      test_probability.flatten()]
    
    value = Dense(1, activation = 'linear')(hidden)
    clf = Dense(10, activation='linear')(hidden)
        
    return gru, logits, value, test_probability, score_probability, clf


def train_model(train_batch, label_batch, test_len=4):
    score    = 0 #save the score 
    test = 0 #question id 
    tests = [] #save question list 
    score_gain    = [] #save score list 
    rewards   = []
    hidden = np.zeros((train_batch.shape[0], train_batch.shape[0]))
    
    for n in range(test_len):
        hidden, logits, value, test, score, _ = policy_model(test,score, hidden, train_batch)
        tests.append(test)
        score_gain.append(score)

        saved_log_probs = []
        saved_values = []
        
        logits, value, hidden, probs, _ = policy_model(0, 0, hidden, train_batch)
        log_probs = np.log(probs)

        for question, point in zip(tests, score_gain):
            log_prob = [i[j][0] for (i, j) in zip(log_probs, question)]
            log_prob = np.matrix(log_prob).reshape(len(log_prob), 1)
            saved_log_probs.append(log_prob)
            saved_values.append(value)
            
            hidden, logits, value, probs, clf_logits, clf = policy_model(question, point, hidden)
            log_probs = np.log(probs)

        loss = tf.keras.losses.CategoricalCrossentropy(clf, label_batch)
        criterion = nn.CrossEntropyLoss(reduce=False)
        
        clf_rewards = []
        for clf_logit, targ in zip(clf_logits, label_batch):
            reward = - criterion(Variable(torch.Tensor(clf_logit.numpy()).unsqueeze(0)), Variable(torch.LongTensor([targ]))).data
            clf_rewards.append(reward.unsqueeze(0))   
        
        clf_rewards = torch.cat(clf_rewards, 0).unsqueeze(-1)     
        rewards.append(clf_rewards)    
        
        # cummulative reward
        R=0
        returns = []
        for r in reversed(rewards):
            R = r + 0.99 * R
        returns.insert(0, R)
        returns = torch.cat(returns, dim=1)
        
        saved_log_probs = torch.cat(saved_log_probs, 1)
        saved_values    = torch.cat(saved_values, 1)
        
        #advtange 
        advantages = Variable(returns) - saved_values
        
        critic_loss = advantages.pow(2).mean()
        actor_loss  = - (saved_log_probs * Variable(advantages.data)).mean()
        optimizer = optim.Adam(policy.parameters())
        optimizer.zero_grad()
        (critic_loss + actor_loss + loss).backward()
        optimizer.step()    

