# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 21:27:57 2018

@author: Auser
"""

''' suppose 0 and 1 '''


def basic(predictor,response):
    TP,TN,FP,FN = 0, 0, 0, 0
    for i in range(len(predictor)):
        if predictor[i] == 1:
            if response[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if response[i] == 0:
                TN += 1
            else:
                FN += 1
    return [TP,TN,FP,FN]



def accuracy(predictor,response):
    tp,tn,fp,fn = basic(predictor,response)
    return (tp+tn)/(tp+tn+fp+fn)
    
    
    
def precision(predictor,response):
    tp,tn,fp,fn = basic(predictor,response)
    return tp/(tp+fp)
    
    

def recall(predictor,response):
    tp,tn,fp,fn = basic(predictor,response)
    return tp/(tp+fn)
    
    
    
def F1score(predictor,response):
    P = precision(predictor,response)
    R = recall(predictor,response)
    return 2*P*R/(P+R)