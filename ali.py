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

# =======================setting=============================
dataset = 'loam'
batch_size = 64
Z = 128
lr = 0.0002
beta1 = 0.5
epochs = 1000
resume = False

def setMod(sym, data_name, ctx, data_iter=None):
    if data_name == "comb":
        mod = mx.mod.Module(symbol=sym, data_names=('data_x', 'data_z',), label_names=('label',), context=ctx)
        mod.bind(data_shapes=[('data_x', (batch_size, 3, 64, 64)), ('data_z', (batch_size, 128, 1, 1))],
                 label_shapes=[('label', (batch_size,))],
                 inputs_need_grad=True)
    else:
        mod = mx.mod.Module(symbol=sym, data_names=(data_name, ), label_names=None, context=ctx)
        mod.bind(data_shapes=data_iter.provide_data)

    if resume:
        _, arg_params, aux_params = mx.model.load_checkpoint(modEC_prefix, modEC_epoch)
        mod.set_params(arg_params, aux_params)
    else:
        mod.init_params(initializer=mx.init.Normal(0.02))
        
    mod.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 0.,
            'beta1': beta1,
        })

    return mod


def main():
    logging.basicConfig(level=logging.DEBUG)
    
    ctx = mx.gpu(3)
    
    
    # =======================symbol==============================
    encoder, decoder, disc = sym_cifar10()
    
    if not resume:
        with open('model/sym_encoder.json',   'wb') as f:
            cPickle.dump(encoder.tojson(), f, cPickle.HIGHEST_PROTOCOL)
        with open('model/sym_decoder.json',   'wb') as f:
            cPickle.dump(decoder.tojson(), f, cPickle.HIGHEST_PROTOCOL)
        with open('model/sym_disc_comb.json', 'wb') as f:
            cPickle.dump(disc.tojson(), f, cPickle.HIGHEST_PROTOCOL)
    else:
        print "read model"

    #mx.viz.plot_network(encoder, shape={'data': (batch_size, 3, 64, 64)}).view()
    #mx.viz.plot_network(decoder, shape={'rand': (batch_size, 128, 1, 1)}).view()
    #mx.viz.plot_network(disc, shape={'data_x': (batch_size, 3, 64, 64), 'data_z': (batch_size, 128, 1, 1)}).view()

    # =======================data================================
    imdb = loamBatch(name='loam').gt_imdb()
    X_train, X_test = get_maps(imdb)
    print 'Train data', X_train.shape[0], ' Test data ', X_test.shape[0]
    train_iter = mx.io.NDArrayIter(X_train, batch_size=batch_size)
    rand_iter  = RandIter(batch_size, Z)
    label      = mx.nd.zeros((batch_size,), ctx=ctx)
    
    # =======================module==============================
    # model encoder
    modEX = setMod(encoder, 'data', ctx, train_iter)
    # model decoder
    modEZ = setMod(decoder, 'rand', ctx, rand_iter)
    # model disc_comb
    modDC = setMod(disc, 'comb',  ctx)
    print "generate model done ..."

    # =======================metric==============================
    def facc(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return -(label*np.log(pred+1e-12) + (1.-label)*np.log(1.-pred+1e-12))
    
    def fentropy(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return -(label*np.log(pred+1e-12) + (1.-label)*np.log(1.-pred+1e-12)).mean()
    
    mEX = mx.metric.CustomMetric(fentropy)
    mEZ = mx.metric.CustomMetric(fentropy)
    mDC = mx.metric.CustomMetric(fentropy)
    mACC = mx.metric.CustomMetric(facc)

    # =======================train===============================
    print 'Training...'
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')

    for epoch in range(epochs):
        train_iter.reset()
        for t, batch in enumerate(train_iter):
            rbatch = rand_iter.next()

            modEZ.forward(rbatch, is_train=True)
            outEZ = modEZ.get_outputs()
            # update discriminator for (fake x, rand z) combination
            label[:] = 0
            modDC.forward(mx.io.DataBatch(outEZ+rbatch.data, [label]), is_train=True)
            modDC.backward()
            gradDC = [[grad.copyto(grad.context) for grad in grads] for grads in modDC._exec_group.grad_arrays]
            modDC.update_metric(mDC, [label])
            modDC.update_metric(mACC, [label])

            # encoder real x into code z
            modEX.forward(batch, is_train=True)
            outEX = modEX.get_outputs()
            # update discriminator for (real x, code z) combination
            label[:] = 1
            modDC.forward(mx.io.DataBatch(batch.data+outEX, [label]), is_train=True)
            modDC.backward()
            for gradsr, gradsf in zip(modDC._exec_group.grad_arrays, gradDC):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr += gradf
                    modDC.update()
            modDC.update_metric(mDC, [label])
            modDC.update_metric(mACC, [label])

            # update decoder
            label[:] = 0
            modDC.forward(mx.io.DataBatch(outEZ+rbatch.data, [label]), is_train=True)
            modDC.backward()
            diffDE = modDC.get_input_grads()[0]
            modEZ.backward([diffDE])
            modEZ.update()
            mEZ.update([label], modDC.get_outputs())

            # update encoder
            label[:] = 1
            modDC.forward(mx.io.DataBatch(batch.data+outEX, [label]), is_train=True)
            modDC.backward()
            diffEN = modDC.get_input_grads()[1]
            modEX.backward([diffEN])
            modEX.update()
            mEX.update([label], modDC.get_outputs())

            t += 1
            if t%10==0:
                print 'epoch:', epoch, ' iter:', t, ' metric:', mACC.get(), mEX.get(), mEZ.get(), mDC.get()
                mACC.reset()
                mEX.reset()
                mEZ.reset()
                mDC.reset()

if __name__ == '__main__':
    main()
