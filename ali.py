import mxnet as mx
import numpy as np
import logging
import cv2
import os
import sys
import cPickle
from datetime import datetime
sys.path.append('vgan')

from data import loamBatch, get_maps, RandIter
from sym_ali import sym_cifar10
from util import fill_buf, visual

logging.basicConfig(level=logging.DEBUG)
    
# =============setting============
batch_size = 64
Z = 100
lr = 0.0002
beta1 = 0.5
ctx = mx.gpu(3)

encoder, decoder, disc_x, disc_z, disc_comb = sym_cifar10()
            
#print symD.tojson()
#mx.viz.plot_network(symG, shape={'rand': (batch_size, 100, 1, 1)}).view()
mx.viz.plot_network(encoder, shape={'data': (batch_size, 3, 64, 64)}).view()
