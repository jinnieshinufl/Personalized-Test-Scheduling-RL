#!/usr/bin/env python
# coding: utf-8

import utils 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

def final_question_set(questions_test, og_test):
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
    
    import pandas as pd 
    
    results = pd.DataFrame(index = og_test.index)
    results['original'] = og_test
    results['ntest'] = results.original.apply(len)
    results = results[:len(recom)]
                
    results['recommend'] = recom            
    results['new_ntest'] = results.recommend.apply(len) 

    return recom, results 

class Policy(nn.Module):
    '''
    ##############################
    # modified from the original code by Nurakhmetov (2019)
    References:
    Nurakhmetov, D. (2019). Reinforcement Learning Applied to Adaptive Classification Testing. 
    In Theoretical and Practical Advances in Computer-based Educational Measurement (pp. 325-336). Springer, Cham.    
    ###############################
    '''
    def __init__(self, n_tests, n_scores, hidden_size):
        super(Policy, self).__init__()
        
        self.t_emb  = nn.Embedding(n_tests, hidden_size // 2)
        self.s_emb  = nn.Embedding(n_scores, hidden_size // 2)
        self.gru    = nn.GRUCell(hidden_size, hidden_size)
        self.actor  = nn.Linear(hidden_size, n_tests)
        self.critic = nn.Linear(hidden_size, 1)
        self.clf    = nn.Linear(hidden_size, 10) 
        self.input_question = nn.Parameter(torch.rand(hidden_size // 2)) 
        self.input_point    = nn.Parameter(torch.rand(hidden_size // 2)) 
        
    def forward(self, test, score, hidden, batch_size=None):
       
        if test is None and score is None and batch_size is not None:
            test_embed = self.input_test.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
            score_embed = self.input_score.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            test_embed = nn.functional.relu(self.t_emb(test))
            score_embed = nn.functional.relu(self.s_emb(score))
        
        embedded = torch.cat([test_embed, score_embed], 2)
        hidden   = self.gru(embedded.squeeze(1), hidden)
        
        logits   = self.actor(hidden)
        value    = self.critic(hidden)
        
        clf_logits = self.clf(hidden)
        return logits, value, hidden, clf_logits

def train(train_batch, train_label, hidden_size, test_length = 5):
    '''
    ##############################
    # modified from the original code by Nurakhmetov (2019)
    references:
    Nurakhmetov, D. (2019). Reinforcement Learning Applied to Adaptive Classification Testing. 
    In Theoretical and Practical Advances in Computer-based Educational Measurement (pp. 325-336). Springer, Cham.    
    ###############################
    '''
    batch_size = 10
    
    tests=[]
    tests_train = [] 
    policy = Policy(n_tests = 18, n_scores= 1401,  hidden_size=hidden_size)
    optimizer = optim.Adam(policy.parameters())
    criterion = nn.CrossEntropyLoss(reduce=False)

    (score, test)  = tuple(None, None)
    tests,  scores =  [], []
    rewards   = []
    hidden = Variable(torch.zeros(batch_size, test), volatile=True)        

    for t in range(test_length):
        logits, value, hidden, _ = policy(test, score, hidden, batch_size)
        probs =nn.functional.softmax(logits)            #sample next item
        next_test = torch.multinomial(probs, 1)

        test = next_test.data.squeeze(1)
        score, test = utils.test_score(train_batch, test)

        masks = []
        for prev_test in tests:
            mask = prev_test.squeeze(1).eq(test).unsqueeze(1)
            masks.append(mask)
        if len(masks) > 0:
            masks = torch.cat(masks, 1)
            masks = masks.sum(1).gt(0)
            masks = -1 * masks.float()
            rewards.append(masks.unsqueeze(1))
        
        tests.append(test.unsqueeze(1))
        scores.append(score.unsqueeze(1))

        score    = Variable(score.unsqueeze(1), volatile=True)
        test = Variable(test.unsqueeze(1), volatile=True)
        
        tests_train.append(tests)

    saved_log_probs = []
    saved_values = []
    
    hidden = Variable(torch.zeros(batch_size, tests))
    logits, value, hidden, _ = policy(None, None, hidden, batch_size)
    log_probs = nn.functional.log_softmax(logits)

    for test, score in zip(test, scores):

        log_prob = log_probs.gather(1, Variable(test))
        saved_log_probs.append(log_prob)
        saved_values.append(value)

        logits, value, hidden, clf_logits = policy(Variable(test), Variable(score), hidden, batch_size)
        log_probs = nn.functional.log_softmax(logits)

    loss = nn.functional.cross_entropy(clf_logits, Variable(train_label))
    
    clf_rewards = []
    for clf_logit, targ in zip(clf_logits.data, train_label):
        reward = - criterion(Variable(clf_logit.unsqueeze(0)), Variable(torch.LongTensor([targ]))).data
        clf_rewards.append(reward.unsqueeze(0))
    clf_rewards = torch.cat(clf_rewards, 0).unsqueeze(-1)
    
    rewards.append(clf_rewards)
    returns = utils.get_reward(rewards)

    saved_log_probs = torch.cat(saved_log_probs, 1)
    saved_values    = torch.cat(saved_values, 1)

    advantages = Variable(returns) - saved_values

    critic_loss = advantages.pow(2).mean()
    actor_loss  = - (saved_log_probs * Variable(advantages.data)).mean()

    optimizer.zero_grad()
    (critic_loss + actor_loss + loss).backward()
    optimizer.step()       
    
    return tests_train
  