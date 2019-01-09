import sklearn.neighbors as nn
import numpy as np
import os
import cv2

def check_value(inds, val):
    ''' Check to see if an array is a single element equaling a particular value
    for pre-processing inputs in a function '''
    if(np.array(inds).size==1):
        if(inds==val):
            return True
    return False

def na(): # shorthand for new axis
    return np.newaxis

def flatten_nd_array(pts_nd,axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd       N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
    pts_flt = pts_nd.transpose((axorder))
    pts_flt = pts_flt.reshape(NPTS,SHP[axis])
    return pts_flt

def unflatten_2d_array(pts_flt,pts_nd,axis=1,squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])

    if(squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        # print NEW_SHP
        # print pts_flt.shape
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out

class NNEncode():
    ''' Encode points using NN search and Gaussian kernel '''
    def __init__(self,NN,sigma,km_filepath='',cc=-1):
        if(check_value(cc,-1)):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = nn.NearestNeighbors(n_neighbors=self.NN, algorithm='ball_tree').fit(self.cc)

        self.alreadyUsed = False

    def encode_points_mtx_nd(self,pts_nd,axis=1,returnSparse=False,sameBlock=True):

        pts_flt = flatten_nd_array(pts_nd,axis=axis)

        P = pts_flt.shape[0]
        if(sameBlock and self.alreadyUsed):
            self.pts_enc_flt[...] = 0 # already pre-allocated
        else:
            self.alreadyUsed = True
            self.pts_enc_flt = np.zeros((P,self.K))
            self.p_inds = np.arange(0,P,dtype='int')[:,na()]

        P = pts_flt.shape[0]
        # print(self.cc.shape)
        (dists,inds) = self.nbrs.kneighbors(pts_flt)

        wts = np.exp(-dists**2/(2*self.sigma**2))
        wts = wts/np.sum(wts,axis=1)[:,na()]

        self.pts_enc_flt[self.p_inds,inds] = wts
        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt,pts_nd,axis=axis)

        return pts_enc_nd

    # def decode_points_mtx_nd(self,pts_enc_nd,axis=1):
    #     pts_enc_flt = flatten_nd_array(pts_enc_nd,axis=axis)
    #     pts_dec_flt = np.dot(pts_enc_flt,self.cc)
    #     pts_dec_nd = unflatten_2d_array(pts_dec_flt,pts_enc_nd,axis=axis)
    #     return pts_dec_nd

    # def decode_1hot_mtx_nd(self,pts_enc_nd,axis=1,returnEncode=False):
    #     pts_1hot_nd = nd_argmax_1hot(pts_enc_nd,axis=axis)
    #     pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd,axis=axis)
    #     if(returnEncode):
    #         return (pts_dec_nd,pts_1hot_nd)
    #     else:
    #         return pts_dec_nd

def encode(data_ab_ss):
  '''Encode to 313bin
  Args:
    data_ab_ss: [N, H, W, 2]
  Returns:
    gt_ab_313 : [N, H, W, 313]
  '''
  NN = 10
  sigma = 5.0
  enc_dir = './'

  data_ab_ss = np.transpose(data_ab_ss[:,::4,::4,:], (0, 3, 1, 2))
  nnenc = NNEncode(NN, sigma, km_filepath=os.path.join(enc_dir, 'myhull.npy'))
  gt_ab_313 = nnenc.encode_points_mtx_nd(data_ab_ss, axis=1)

  gt_ab_313 = np.transpose(gt_ab_313, (0, 2, 3, 1))
  return gt_ab_313

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1) # only difference

def decode(data_l, conv8_313, rebalance=1,need_softmax = True):
    """
    Args:
      data_l   : [1, height, width, 1]
      conv8_313: [1, heiht/4, width/4, 313]
    Returns:
      img_rgb  : [height, width, 3]
    """
    data_l = data_l # + 50
    _, height, width, _ = data_l.shape
    data_l = data_l[0, :, :, :]
    conv8_313 = conv8_313[0, :, :, :]
    enc_dir = './'
    conv8_313_rh = conv8_313 * rebalance

    class8_313_rh = softmax(conv8_313_rh) if need_softmax else conv8_313_rh


    cc = np.load(os.path.join(enc_dir, 'myhull.npy'))
    data_ab = np.dot(class8_313_rh, cc)

    # data_ab = cc[class8_313_rh.argmax(axis=-1)].astype(np.float32)
    data_ab = cv2.resize(data_ab, (height, width))

    img_lab = np.concatenate((data_l, data_ab), axis=-1)
    img_lab = img_lab.astype(dtype=np.uint8)
    return img_lab

if __name__ == "__main__":
    images = np.load("images.npy")
    images = images[5:6]
    test = images[:,:,:,1:]
    # for i in range(200):
    #     for j in range(200):
    #         test[0][j][i][0] = 40
    ab = encode(test)
    # print(ab[0][0][0])
    x = decode(images[0:1,:,:,0:1], ab[0:1,:,:,:], 1, False)
    x = cv2.cvtColor(x, cv2.COLOR_LAB2BGR)
    cv2.imwrite("res.png", x)


