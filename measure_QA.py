#-*- coding: utf-8 -*-

"""
re-implement wang's code
https://github.com/shuohangwang/SeqMatchSeq/blob/master/util/utils.lua
"""
import numpy as np

"""
y_true  : label
y_score: predicted value
must -> len(y_true) == len(y_score)
"""
def MAP(y_true, y_score):

    ground_label = y_true
    predict_label = []

    map = float(0)
    map_idx = 0
    extracted = []

    for i in range( len(ground_label) ):
        if ground_label[i] != 0 :
            extracted.append(i)

    sss = sorted( y_score, reverse=True )
    sss_rank = [ list( y_score).index(x) for x in sss]
    predict_label = np.asarray(sss_rank)
    
    #print predict_label
    #print extracted
    
    for i in range( len(predict_label) ) :
        if predict_label[i] in extracted :
            map_idx = map_idx + 1
            map = map + map_idx / float(i+1)

    if map_idx != 0:
        map = map / float(map_idx)
    else:
        map = 0
    
    return map


def MRR(y_true, y_score):

    ground_label = y_true
    predict_label = []

    mrr = float(0)
    mrr_idx = 0
    extracted = []

    for i in range( len(ground_label) ):
        if ground_label[i] != 0 :
            extracted.append(i)

    sss = sorted( y_score, reverse=True )
    sss_rank = [ list( y_score).index(x) for x in sss]
    predict_label = np.asarray(sss_rank)
    
    #print predict_label
    #print extracted
    
    for i in range( len(predict_label) ) :
        if predict_label[i] in extracted :
            mrr = 1.0 / float(i+1)
            break

    return mrr