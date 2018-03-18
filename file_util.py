#-*- coding: utf-8 -*-


import os
import chardet

'''
dirname        : path that need to be searched
ret                : files in the dirname (recursive)
list_avoid_dir : dirname need to be skipped
usage           : 
    list_files = []
    file_search(dirname, list_files):   
'''
def file_search(dirname, ret, list_avoid_dir=[]):
    
    filenames = os.listdir(dirname)
    
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)

        if os.path.isdir(full_filename) :
            if full_filename.split('/')[-1] in list_avoid_dir:
                continue
            else:
                file_search(full_filename, ret, list_avoid_dir)
            
        else:
            ret.append( full_filename )          

            

'''
filename : filename (inc. path) that will be inspected
'''
def find_encoding(filename):
    rawdata = open(filename, 'rb').read()
    result = chardet.detect(rawdata)
    charenc = result['encoding']    
    return charenc
            