import mxnet as mx

def conv_net(data, name, kernel, stride, filter, bn=True, act_type = "leaky", pad=(1,1), drop=-1.0, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    BatchNorm = mx.sym.BatchNorm
    conv = mx.sym.Convolution(data, name=name+'_conv', kernel=kernel, stride=stride, pad=pad, num_filter=filter, no_bias=no_bias)

    if bn:
        norm = BatchNorm(conv, name=name+'_bn', fix_gamma=fix_gamma, eps=eps)
    else:
        norm = conv
        
    if act_type == "leaky":
        act = mx.sym.LeakyReLU(norm, name=name+'_act', act_type='leaky', slope=0.2)
    else:
        act = mx.sym.Activation(norm, name=name+'_act', act_type=act_type)

    return act
        
def deconv_net(data, name, kernel, stride, filter, bn=False, act_type = "leaky", pad=(1,1), no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    BatchNorm = mx.sym.BatchNorm
    conv = mx.sym.Deconvolution(data, name=name+'_conv', kernel=kernel, stride=stride, pad=pad, num_filter=filter, no_bias=no_bias)

    if bn:
        norm = BatchNorm(conv, name=name+'_bn', fix_gamma=fix_gamma, eps=eps)
    else:
        norm = conv
    
    if act_type == "leaky":
        act = mx.sym.LeakyReLU(norm, name=name+'_act', act_type='leaky', slope=0.2)
    else:
        act = mx.sym.Activation(norm, name=name+'_act', act_type=act_type)

    return act

def sym_cifar10(no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):

    BatchNorm = mx.sym.BatchNorm

    ##=============Encoder==============##
    data = mx.sym.Variable('data') # 3x64x64 input
    gz1 = conv_net(data, 'gz1', kernel=(4,4), stride=(2,2), pad=(1,1), filter=32, bn=True)
    gz2 = conv_net(gz1,  'gz2', kernel=(4,4), stride=(2,2), pad=(1,1), filter=64, bn=True)
    gz3 = conv_net(gz2,  'gz3', kernel=(4,4), stride=(2,2), pad=(1,1), filter=128, bn=True)
    gz4 = conv_net(gz3,  'gz4', kernel=(4,4), stride=(2,2), pad=(1,1), filter=256, bn=True)
    gz5 = conv_net(gz4,  'gz5', kernel=(4,4), stride=(2,2), pad=(1,1), filter=512, bn=True)
    gz6 = conv_net(gz5,  'gz6', kernel=(4,4), stride=(2,2), pad=(1,1), filter=512, bn=True)
    gz7 = conv_net(gz6,  'gz7', kernel=(1,1), stride=(1,1), pad=(0,0), filter=128, bn=False, act_type="relu")
    encoder = gz7

    ##=============Decoder==============##
    rand = mx.sym.Variable('rand') # 128x1x1 input
    gx1 = deconv_net(rand, 'gx1', kernel=(4,4), stride=(2,2), filter=512, bn=True)
    gx2 = deconv_net(gx1,  'gx2', kernel=(4,4), stride=(2,2), filter=512, bn=True)
    gx3 = deconv_net(gx2,  'gx3', kernel=(4,4), stride=(2,2), filter=256, bn=True)
    gx4 = deconv_net(gx3,  'gx4', kernel=(4,4), stride=(2,2), filter=128, bn=True)
    gx5 = deconv_net(gx4,  'gx5', kernel=(4,4), stride=(2,2), filter=64 , bn=True)
    gx6 = deconv_net(gx5,  'gx6', kernel=(4,4), stride=(2,2), filter=32 , bn=True)
    gx7 = deconv_net(gx6,  'gx7', kernel=(1,1), stride=(1,1), pad=(0,0), filter=3  , bn=False, act_type="sigmoid")
    decoder = gx7

    ##========Discrimator x ============##
    data_x = mx.sym.Variable('data_x') # 3x64x64 input, 512 output
    dx1 = conv_net(data_x, 'dx1', kernel=(4,4), stride=(2,2), filter=32,  bn=False, drop=0.2, act_type="softrelu")
    dx2 = conv_net(dx1,  'dx2', kernel=(4,4), stride=(2,2), filter=64,  bn=False, drop=0.5, act_type="softrelu")
    dx3 = conv_net(dx2,  'dx3', kernel=(4,4), stride=(2,2), filter=128, bn=False, drop=0.5, act_type="softrelu")
    dx4 = conv_net(dx3,  'dx4', kernel=(4,4), stride=(2,2), filter=256, bn=False, drop=0.5, act_type="softrelu")
    dx5 = conv_net(dx4,  'dx5', kernel=(4,4), stride=(2,2), filter=512, bn=False, drop=0.5, act_type="softrelu")
    dx6 = conv_net(dx5,  'dx6', kernel=(4,4), stride=(2,2), filter=512, bn=False, drop=0.5, act_type="softrelu")

    ##========Discrimator z ============##
    data_z = mx.sym.Variable('data_z') # 128x1x1 input, 512 output
    dz1 = conv_net(data_z, 'dz1', kernel=(1,1), stride=(1,1), pad=(0,0), filter=512, bn=False, drop=0.2, act_type="softrelu")
    dz2 = conv_net(dz1,    'dz2', kernel=(1,1), stride=(1,1), pad=(0,0), filter=512, bn=False, drop=0.5, act_type="softrelu")

    ##========Discrimator x, z ============##
    label = mx.sym.Variable('label')
    comb = mx.sym.Concat(dx6, dz2, dim=1)
    disc1 = conv_net(comb,  'dxz1', kernel=(1,1), stride=(1,1), pad=(0,0), filter=1024, bn=False, drop=0.5, act_type="softrelu")
    disc2 = conv_net(disc1, 'dxz2', kernel=(1,1), stride=(1,1), pad=(0,0), filter=1024, bn=False, drop=0.5, act_type="softrelu")
    disc3 = conv_net(disc2, 'dxz3', kernel=(1,1), stride=(1,1), pad=(0,0), filter=1,    bn=False, drop=0.5, act_type="softrelu")
    disc3 = mx.sym.Flatten(disc3)

    disc_loss = mx.sym.LogisticRegressionOutput(data=disc3, label=label, name='disc_loss')
    return encoder, decoder, disc_loss

def sym_loam(nen, nde, ndx, ndz, ndisc, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):

    BatchNorm = mx.sym.BatchNorm

    ##=============Encoder==============##
    data = mx.sym.Variable('data') # 3x64x64 input
    gz1 = conv_net(data, 'gz1', kernel=(4,4), stride=(2,2), pad=(1,1), filter=nen,   bn=True)
    gz2 = conv_net(gz1,  'gz2', kernel=(4,4), stride=(2,2), pad=(1,1), filter=nen*2, bn=True)
    gz3 = conv_net(gz2,  'gz3', kernel=(4,4), stride=(2,2), pad=(1,1), filter=nen*4, bn=True)
    gz4 = conv_net(gz3,  'gz4', kernel=(4,4), stride=(2,2), pad=(1,1), filter=nen*8, bn=True)
    gz5 = conv_net(gz4,  'gz5', kernel=(4,4), stride=(2,2), pad=(1,1), filter=nen, bn=True)
    gz6 = conv_net(gz5,  'gz6', kernel=(4,4), stride=(2,2), pad=(1,1), filter=512, bn=True)
    gz7 = conv_net(gz6,  'gz7', kernel=(1,1), stride=(1,1), pad=(0,0), filter=128, bn=False, act_type="relu")
    encoder = gz7

    ##=============Decoder==============##
    rand = mx.sym.Variable('rand') # 128x1x1 input
    gx1 = deconv_net(rand, 'gx1', kernel=(4,4), stride=(2,2), filter=512, bn=True)
    gx2 = deconv_net(gx1,  'gx2', kernel=(4,4), stride=(2,2), filter=512, bn=True)
    gx3 = deconv_net(gx2,  'gx3', kernel=(4,4), stride=(2,2), filter=256, bn=True)
    gx4 = deconv_net(gx3,  'gx4', kernel=(4,4), stride=(2,2), filter=128, bn=True)
    gx5 = deconv_net(gx4,  'gx5', kernel=(4,4), stride=(2,2), filter=64 , bn=True)
    gx6 = deconv_net(gx5,  'gx6', kernel=(4,4), stride=(2,2), filter=32 , bn=True)
    gx7 = deconv_net(gx6,  'gx7', kernel=(1,1), stride=(1,1), pad=(0,0), filter=3  , bn=False, act_type="sigmoid")
    decoder = gx7

    ##========Discrimator x ============##
    data_x = mx.sym.Variable('data_x') # 3x64x64 input, 512 output
    dx1 = conv_net(data_x, 'dx1', kernel=(4,4), stride=(2,2), filter=32,  bn=False, drop=0.2, act_type="softrelu")
    dx2 = conv_net(dx1,  'dx2', kernel=(4,4), stride=(2,2), filter=64,  bn=False, drop=0.5, act_type="softrelu")
    dx3 = conv_net(dx2,  'dx3', kernel=(4,4), stride=(2,2), filter=128, bn=False, drop=0.5, act_type="softrelu")
    dx4 = conv_net(dx3,  'dx4', kernel=(4,4), stride=(2,2), filter=256, bn=False, drop=0.5, act_type="softrelu")
    dx5 = conv_net(dx4,  'dx5', kernel=(4,4), stride=(2,2), filter=512, bn=False, drop=0.5, act_type="softrelu")
    dx6 = conv_net(dx5,  'dx6', kernel=(4,4), stride=(2,2), filter=512, bn=False, drop=0.5, act_type="softrelu")

    ##========Discrimator z ============##
    data_z = mx.sym.Variable('data_z') # 128x1x1 input, 512 output
    dz1 = conv_net(data_z, 'dz1', kernel=(1,1), stride=(1,1), pad=(0,0), filter=512, bn=False, drop=0.2, act_type="softrelu")
    dz2 = conv_net(dz1,    'dz2', kernel=(1,1), stride=(1,1), pad=(0,0), filter=512, bn=False, drop=0.5, act_type="softrelu")

    ##========Discrimator x, z ============##
    label = mx.sym.Variable('label')
    comb = mx.sym.Concat(dx6, dz2, dim=1)
    disc1 = conv_net(comb,  'dxz1', kernel=(1,1), stride=(1,1), pad=(0,0), filter=1024, bn=False, drop=0.5, act_type="softrelu")
    disc2 = conv_net(disc1, 'dxz2', kernel=(1,1), stride=(1,1), pad=(0,0), filter=1024, bn=False, drop=0.5, act_type="softrelu")
    disc3 = conv_net(disc2, 'dxz3', kernel=(1,1), stride=(1,1), pad=(0,0), filter=1,    bn=False, drop=0.5, act_type="softrelu")
    disc3 = mx.sym.Flatten(disc3)

    disc_loss = mx.sym.LogisticRegressionOutput(data=disc3, label=label, name='disc_loss')
    return encoder, decoder, disc_loss
