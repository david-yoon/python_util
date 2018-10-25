#-*- coding: utf-8 -*-

import numpy as np

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
    #sss_rank = [ list( y_score).index(x) for x in sss]
    
    # for resolving same number
    sss_rank = []
    for x in sss:
        list_index = ([i for i, val in enumerate(y_score) if val==x])
        if( len(list_index) == 1 ):
            sss_rank.append( list_index[0] )
        else:
            for i in list_index:
                if i not in sss_rank:
                    sss_rank.append(i)
                    break    
    
    
    predict_label = np.asarray(sss_rank)
    
    #print predict_label
    #print extracted
    
    for i in range( len(predict_label) ) :
        if predict_label[i] in extracted :
            mrr = 1.0 / float(i+1)
            break

    return mrr