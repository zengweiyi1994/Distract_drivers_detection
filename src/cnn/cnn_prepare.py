#!/usr/bin/env python

import random
import csv
import os

TRAIN_DIR = './dataset/train'
TEST_DIR = './dataset/test'
META_FILE = './dataset/driver_imgs_list.csv'

'''
'''
SAMPLE_SIZE = 20

def file_list(input_dir, extension):
    flist = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f[-4:] == '.' + extension:
                flist.append(os.path.join(root, f))
    return flist

def encode_label(labels):
    return {l:i for i,l in enumerate(labels)}

def load_label_map():
    label_map = {}
    csv_data = csv.reader(open(META_FILE))
    row_num = 0
    encoded_label = []
    for row in csv_data:
        if row_num == 0:
            tags = row
        else:
            label_map[row[2]] = row[1]
        row_num = row_num + 1
    encoded_label = encode_label(set(label_map.values()))
    print "Encoded label: %s" % str(encoded_label)
    return { f:encoded_label[l] for f, l in label_map.iteritems() }

def select(flist, label_map, n):
    inv_map = {}
    for f in flist:
        k = label_map[os.path.basename(f)]
        inv_map.setdefault(k, []).append(f)
    return [v for k in inv_map for v in random.sample(inv_map[k], n) ]

def prepare_train_files(sample=False, shuffle=True):
    label_map = load_label_map()
    flist = file_list(TRAIN_DIR, 'jpg')
    if sample:
        flist = select(flist, label_map, SAMPLE_SIZE)
    if shuffle:
        flist = random.shuffle(flist)
    labels = [label_map[os.path.basename(f)] for f in flist]
    return flist, labels

def prepare_test_files(sample=False, shuffle=True):
    flist = file_list(TEST_DIR, 'jpg')
    if sample:
        flist = random.sample(flist, SAMPLE_SIZE*3)
    if shuffle:
        flist = random.shuffle(flist)
    return flist

if __name__ == '__main__':
    print 'Load training data...'
    flist, labels = prepare_train_files()
    print len(flist), len(labels)

    print 'Load testing data...'
    flist = prepare_test_files()
    print len(flist)
