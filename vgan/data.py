import mxnet as mx
import numpy as np
import os
import cPickle
import cv2

def get_maps(imdb):
    maps = imdb
    np.random.seed(1234)
    p = np.random.permutation(len(maps))
    X = np.array(maps)
    X = X[p]
    X = np.asarray([cv2.resize(x, (64, 64)) for x in X])

    X = X.astype(np.float32)/(255.0/2) - 1.0
    X = X.reshape((-1, 1, 64, 64))
    X = np.tile(X, (1, 3, 1, 1))
    X_train = X[:800]
    X_test  = X[800:]

    return X_train, X_test

class loamBatch(object):
    def __init__(self, name, pad=0):
        self.name = name
        self.pad = pad

    def gt_imdb(self):
        cache_file = 'data/cache/img.pkl'
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                imdb = cPickle.load(f)
                print '{} gt imdb loaded from {}'.format(self.name, cache_file)
                return imdb
        gt_imdb = self.load_img()
        with open(cache_file, 'wb') as f:
            cPickle.dump(gt_imdb, f, cPickle.HIGHEST_PROTOCOL)
        return gt_imdb

    def load_img(self):
        img_list = 'data/loam/imglists/img.list'
        assert os.path.exists(img_list), 'Path dose not exist: {}'.format(img_list)
        gt_imdb = []
        with open(img_list, 'r') as f:
            for line in f:
                img_file = os.path.join('data/loam/', 'images', line.strip().split('.')[0] + '.jpg')
                print img_file
                img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                gt_imdb.append(img)
        return gt_imdb

class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim, 1, 1))]

class ImagenetIter(mx.io.DataIter):
    def __init__(self, path, batch_size, data_shape):
        self.internal = mx.io.ImageRecordIter(
            path_imgrec = path,
            data_shape  = data_shape,
            batch_size  = batch_size,
            rand_crop   = True,
            rand_mirror = True,
            max_crop_size  = 256,
            min_crop_size  = 192)
        self.provide_data  = [('data', (batch_size, ) + data_shape)]
        self.provide_label = []

    def reset(self):
        self.internal.reset()
        
    def iter_next(self):
        return self.internal.iternal.iter_next()

    def getdata(self):
        data = self.internal.getdata()
        data = data*(2.0/255.0)
        data -= 1
        return [data]
