#-*- coding: utf-8 -*-

import numpy as np

def MAP(y_true, y_score):

    ground_label = y_true
    predict_label = []

    map = float(0)
    map_idx = 0
    extracted = []

    for i in range( len(ground_label) ):
        if ground_label[i] != 0 :
            extracted.append(i)

            
    # for resolving same number
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
    
#     print (extracted)
#     print (predict_label)
    
    for i in range( len(predict_label) ) :
        if predict_label[i] in extracted :
            map_idx = map_idx + 1
            map = map + map_idx / float(i+1)

    if map_idx != 0:
        map = map / float(map_idx)
    else:
        map = 0
    
    return map