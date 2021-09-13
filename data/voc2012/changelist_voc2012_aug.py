import os
import numpy as np

read_filepath = '/public/home/rfhe/data/voc2012/list/test.txt'
save_filepath = '/public/home/rfhe/data/voc2012/list/test_our.txt'

with open(read_filepath, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        w1= open(save_filepath, 'a')
        # w1.write('JPEGImages/'+line+'.jpg SegmentationClassAug/'+line+'.png \n')
        w1.write('test/JPEGImages/'+line+'.jpg \n')
        # w1.write(line)

