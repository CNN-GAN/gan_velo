import mxnet as mx
import numpy as np
import logging
import cv2
import os
import sys
from datetime import datetime
sys.path.append('vgan')

from data import loamBatch, get_maps, RandIter
from sym import sym_D
logging.basicConfig(level=logging.DEBUG)        
# =============setting============
dataset = 'loam'
imgnet_path = './train.rec'
ndf = 64
ctx = mx.gpu(0)
modD_prefix = 'model/D'
modD_epoch  = 1
batch_size  = 1
symD = sym_D(ndf)

#mx.viz.plot_network(symD, shape={'data': (batch_size, 3, 64, 64)}).view()

# =======================data================================
imdb = loamBatch(name='loam').gt_imdb()
_, X_test = get_maps(imdb, random=False)
test_iter = mx.io.NDArrayIter(X_test, batch_size=1)

# ====================module D==============================
modD = mx.mod.Module(symbol=symD, data_names=('data',), context=ctx)
modD.bind(data_shapes=test_iter.provide_data)

# ===================Load prefix===========================
sym, arg_params, aux_params = mx.model.load_checkpoint(modD_prefix, modD_epoch)
modD.set_params(arg_params, aux_params)

# ===================Estimate Feature=======================
for t, batch in enumerate(test_iter):
    modD.forward(batch, is_train=False)
    feature = str(modD.get_outputs()[0].asnumpy().reshape(1, -1))
    feature = feature[2:-2]

    file_feature = open('data/loam/results/%04d.txt'%t, 'w')
    file_feature.write(feature + '\n')
    file_feature.close()
