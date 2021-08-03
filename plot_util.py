import matplotlib.pyplot as plt
import numpy as np


'''
hitogram + cumulative plot
list_x : keys 
lixt_y : values
'''
def show_hist_cum_plot(list_x, list_y, minN=0, maxN=1000000, p_type=None):

    fig = plt.figure()

    ax1 = fig.add_axes([0,0,2,1])
    ax1.set_ylabel('count', color='navy')
    
    if p_type=='bar':
        ax1.bar(list_x[minN:maxN], list_y[minN:maxN])
    else:
        ax1.plot(list_x[minN:maxN], list_y[minN:maxN])
    ax1.tick_params(axis='y', labelcolor='navy')
    ax1.set_xticklabels(ax1.get_xticks(), rotation = 45)

    norm_cnt = (list_y / np.sum(list_y)).tolist()
    cum_cnt = np.cumsum(norm_cnt)

    ax2 = ax1.twinx()
    ax2.set_ylabel('%', color='red')
    ax2.plot(list_x[minN:maxN], cum_cnt[minN:maxN], color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.xticks(rotation = 45) 
    plt.show()
    

'''
hitogram + cumulative plot
list_data : 
'''
def show_hist_cum_plot_bucketing(list_data, step=1, minN=0, maxN=1000000, p_type=None):
    
    dic_step = {}
    min_val = np.min(list_data)
    max_val = np.max(list_data)
    step = 1
    print(min_val, max_val)

    for i in range( int(max_val/step) +1 ):
        dic_step[i] = 0

    for i in list_data:
        chunk = int(i/step)
        dic_step[chunk] += 1   
    
    show_hist_cum_plot([x*step for x in dic_step.keys()], list(dic_step.values()), minN=minN, maxN=maxN, p_type=p_type)
    
    
