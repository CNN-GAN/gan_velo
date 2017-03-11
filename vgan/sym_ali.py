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
    gz7 = conv_net(gz6,  'gz7', kernel=(1,1), stride=(1,1), filter=128, bn=False, act_type="relu")
    encoder = gz7

    ##=============Decoder==============##
    rand = mx.sym.Variable('rand') # 64x1x1 input
    gx1 = deconv_net(rand, 'gx1', kernel=(4,4), stride=(1,1), filter=256, bn=True)
    gx2 = deconv_net(gx1,  'gx2', kernel=(4,4), stride=(2,2), filter=128, bn=True)
    gx3 = deconv_net(gx2,  'gx3', kernel=(4,4), stride=(1,1), filter=64 , bn=True)
    gx4 = deconv_net(gx3,  'gx4', kernel=(4,4), stride=(2,2), filter=32 , bn=True)
    gx5 = deconv_net(gx4,  'gx5', kernel=(5,5), stride=(1,1), filter=32 , bn=True)
    gx6 =   conv_net(gx5,  'gx6', kernel=(1,1), stride=(1,1), filter=32 , bn=True)
    gx7 =   conv_net(gx6,  'gx7', kernel=(1,1), stride=(1,1), filter=3  , bn=False, act_type="sigmoid")
    decoder = gx7

    ##========Discrimator x ============##
    data = mx.sym.Variable('data') # 3x64x64 input
    dx1 = conv_net(data, 'dx1', kernel=(5,5), stride=(1,1), filter=32,  bn=False, drop=0.2, act_type="softrelu")
    dx2 = conv_net(dx1,  'dx2', kernel=(4,4), stride=(2,2), filter=64,  bn=False, drop=0.5, act_type="softrelu")
    dx3 = conv_net(dx2,  'dx3', kernel=(4,4), stride=(1,1), filter=128, bn=False, drop=0.5, act_type="softrelu")
    dx4 = conv_net(dx3,  'dx4', kernel=(4,4), stride=(2,2), filter=256, bn=False, drop=0.5, act_type="softrelu")
    dx5 = conv_net(dx4,  'dx5', kernel=(4,4), stride=(1,1), filter=512, bn=False, drop=0.5, act_type="softrelu")
    disc_x = dx5

    ##========Discrimator z ============##
    rand = mx.sym.Variable('rand') # 3x64x64 input
    dz1 = conv_net(rand, 'dz1', kernel=(1,1), stride=(1,1), filter=512, bn=False, drop=0.2, act_type="softrelu")
    dz2 = conv_net(dz1,  'dz2', kernel=(1,1), stride=(1,1), filter=512, bn=False, drop=0.5, act_type="softrelu")
    disc_z = dz2

    ##========Discrimator x, z ============##
    combine = mx.sym.Variable('combine') # 3x64x64 input
    dxz1 = conv_net(combine, 'dxz1', kernel=(1,1), stride=(1,1), filter=1024, bn=False, drop=0.5, act_type="softrelu")
    dxz2 = conv_net(dxz1,    'dxz2', kernel=(1,1), stride=(1,1), filter=1024, bn=False, drop=0.5, act_type="softrelu")
    dxz3 = conv_net(dxz2,    'dxz3', kernel=(1,1), stride=(1,1), filter=1,    bn=False, drop=0.5, act_type="softrelu")
    disc_comb = dxz3

    return encoder, decoder, disc_x, disc_z, disc_comb
