#-*- coding: utf-8 -*-


'''
print stats such as mean, max, std
data        : list(numpy) of data
'''
def print_stats(data):
    print("\tmean:\t\t" + '{:.3f}'.format(np.average(data)))
    print("\tstd:\t\t"  + '{:.3f}'.format(np.std(data)))
    print("\tmax:\t\t"  + '{:.3f}'.format(np.max(data)))
    print("\t95.xx coverage: " + '{:.3f}'.format(np.average(data) +  2*np.std(data)))
    print("\t99.73 coverage: " + '{:.3f}'.format(np.average(data) +  3*np.std(data)))
    print("\t99.95 coverage: " + '{:.3f}'.format(np.average(data) +  3.5*np.std(data)))
    print("\t99.99 coverage: " + '{:.3f}'.format(np.average(data) +  4*np.std(data)))
    print('')