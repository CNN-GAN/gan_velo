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
from sym import make_dcgan_sym
from util import fill_buf, visual

def main():

    logging.basicConfig(level=logging.DEBUG)
    
    # =============setting============
    dataset = 'loam'
    imgnet_path = './train.rec'
    ndf = 64
    ngf = 64
    nc = 3
    batch_size = 64
    Z = 100
    lr = 0.0002
    beta1 = 0.5
    ctx = mx.gpu(3)
    check_point = True
    resume = False
    modG_prefix = 'model/G'
    modD_prefix = 'model/D'
    modG_epoch  = 1
    modD_epoch  = 1

    symG, symD = make_dcgan_sym(ngf, ndf, nc)
    if not resume:
        with open('model/symD.json', 'wb') as f:
            cPickle.dump(symD.tojson(), f, cPickle.HIGHEST_PROTOCOL)
            with open('model/symG.json', 'wb') as f:
                cPickle.dump(symG.tojson(), f, cPickle.HIGHEST_PROTOCOL)
            else:
                print 'Load model G from ', modG_prefix, ' epoch ', modG_epoch
                print 'Load model D from ', modD_prefix, ' epoch ', modD_epoch

    #print symD.tojson()
    #mx.viz.plot_network(symG, shape={'rand': (batch_size, 100, 1, 1)}).view()
    #mx.viz.plot_network(symD, shape={'data': (batch_size, nc, 64, 64)}).view()

    # =======================data================================
    imdb = loamBatch(name='loam').gt_imdb()
    X_train, X_test = get_maps(imdb)
    print 'Train data ', X_train.shape[0], ' Test data ', X_test.shape[0]

    train_iter = mx.io.NDArrayIter(X_train, batch_size=batch_size)
    rand_iter  = RandIter(batch_size, Z)
    label      = mx.nd.zeros((batch_size,), ctx=ctx)

    # =====================module G==============================
    modG= mx.mod.Module(symbol=symG, data_names=('rand',), label_names=None, context=ctx)
    modG.bind(data_shapes=rand_iter.provide_data)
    if resume:
        sym, arg_params, aux_params = mx.model.load_checkpoint(modG_prefix, modG_epoch)
        modG.set_params(arg_params, aux_params)
    else:
        modG.init_params(initializer=mx.init.Normal(0.02))

    modG.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 0.,
            'beta1': beta1,
        })
    mods = [modG]

    # ====================module D==============================
    modD = mx.mod.Module(symbol=symD, data_names=('data',), label_names=('label',), context=ctx)
    modD.bind(data_shapes=train_iter.provide_data,
              label_shapes=[('label', (batch_size,))],
              inputs_need_grad=True)
    if resume:
        sym, arg_params, aux_params = mx.model.load_checkpoint(modD_prefix, modD_epoch)
        modD.set_params(arg_params, aux_params)
    else:
        modD.init_params(initializer=mx.init.Normal(0.02))

    modD.init_optimizer(
        optimizer='sgd',
        optimizer_params={
            'learning_rate': lr,
            'wd': 0.,
            'beta1': beta1,
        })
    mods.append(modD)

    # =================printing===================
    def norm_stat(d):
        return mx.nd.norm(d)/np.sqrt(d.size)
        mon = mx.mon.Monitor(10, norm_stat, pattern=".*output|d1_backward_data", sort=True)
        mon = None
        if mon is not None:
            for mod in mods:
                pass

    def facc(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return -(label*np.log(pred+1e-12) + (1.-label)*np.log(1.-pred+1e-12))

    def fentropy(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return -(label*np.log(pred+1e-12) + (1.-label)*np.log(1.-pred+1e-12)).mean()

    mG = mx.metric.CustomMetric(fentropy)
    mD = mx.metric.CustomMetric(fentropy)
    mACC = mx.metric.CustomMetric(facc)

    print 'Training...'
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')

    # ==================train======================
    for epoch in range(1000):
        train_iter.reset()
        for t, batch in enumerate(train_iter):
            rbatch = rand_iter.next()

            if mon is not None:
                mon.tic()

            modG.forward(rbatch, is_train=True)
            outG = modG.get_outputs()

            # update discriminator on fake
            label[:] = 0
            modD.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            modD.backward()
            gradD = [[grad.copyto(grad.context) for grad in grads] for grads in modD._exec_group.grad_arrays]
            
            modD.update_metric(mD, [label])
            modD.update_metric(mACC, [label])

            # update discriminator on real
            label[:] = 1
            batch.label = [label]
            modD.forward(batch, is_train=True)
            modD.backward()
            for gradsr, gradsf in zip(modD._exec_group.grad_arrays, gradD):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr += gradf
                    modD.update()

            modD.update_metric(mD, [label])
            modD.update_metric(mACC, [label])

            # update generator
            label[:] = 1
            modD.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            modD.backward()
            diffD = modD.get_input_grads()
            modG.backward(diffD)
            modG.update()
            
            mG.update([label], modD.get_outputs())


            if mon is not None:
                mon.toc_print()

            t += 1
            if t % 10 == 0:
                print 'epoch:', epoch, 'iter:', t, 'metric:', mACC.get(), mG.get(), mD.get()
                mACC.reset()
                mG.reset()
                mD.reset()

                visual('gout', outG[0].asnumpy())
                diff = diffD[0].asnumpy()
                diff = (diff - diff.mean())/diff.std()
                visual('diff', diff)
                visual('data', batch.data[0].asnumpy())

        if epoch%50 == 0:
            print 'Saving...'
            modG.save_params('model/%s_G_%s-%04d.params'%(dataset, stamp, epoch))
            modD.save_params('model/%s_D_%s-%04d.params'%(dataset, stamp, epoch))


if __name__ == '__main__':
    main()
