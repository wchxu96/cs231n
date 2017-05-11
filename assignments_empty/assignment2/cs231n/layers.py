import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #x = np.reshape(N,-1)
  #out = np.maximum(np.dot(x,w) + b,0)
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  #pass
  N = x.shape[0]
  #print (x.shape)
  x_copy = x.reshape(N,-1)
  #out = np.maximum(np.dot(x,w) + b,0)
  out = np.dot(x_copy,w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  #pass
  shape = x.shape
  #print (shape)
  N = shape[0]
  x = x.reshape(N,-1)
  db = np.sum(dout,axis=0)
  dw = np.dot(x.T,dout)
  dx = np.dot(dout,w.T).reshape(shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return 	dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  #pass
  out = np.maximum(x,0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  #pass
  dout_copy = dout.copy()
  dout_copy[x < 0] = 0
  dx = dout_copy
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    #pass
    #running_mean = momentum * running_mean + (1 - momentum) * np.mean(x,axis=0)
    #running_var  = momentum * running_var + (1 - momentum) * (np.std(x,axis=0) ** 2)
    '''
    mean = np.mean(x,axis=0)
    var = np.std(x,axis=0) ** 2
    out = (x - mean) / (np.sqrt(var) + eps)
    out = out * gamma + beta
    #bn_param['running_mean'] = running_mean
    #bn_param['running_var'] = running_var
    cache = (out,beta,gamma,mean,var,eps)
    running_mean = running_mean * momentum + (1 - momentum) * mean
    running_var = running_mean * momentum + (1 - momentum) * var
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    '''
    N,D = x.shape
    #print x.shape
    mu = 1./N * np.sum(x,axis=0)
    xmu = x - mu
    sq = xmu ** 2
    var = 1./ N * np.sum(sq,axis=0)
    sqrtvar = np.sqrt(var + eps)
    ivar = 1. / sqrtvar
    xhat = ivar * xmu
    gammax = xhat * gamma
    out = gammax + beta
    running_mean = running_mean * momentum + mu * (1 - momentum)
    running_var = running_var * momentum + var * (1 - momentum)
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    #pass
    out = (x - running_mean) / (np.sqrt(running_var + eps))
    out = gamma * out + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  #pass
  #out,beta,gamma,mean,var,eps = cache
  xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
  N,D = dout.shape
  dbeta = np.sum(dout,axis=0)
  dgammax = dout
  dgamma = np.sum(dgammax * xhat,axis=0)
  dxhat = dgammax * gamma
  divar = np.sum(dxhat * xmu,axis=0)
  dxmu1 = dxhat * ivar
  dsqrtvar = -1. / (sqrtvar ** 2) * divar
  dvar = 1./(2 * np.sqrt(var + eps)) * dsqrtvar
  dsq = np.ones((N,D)) * 1./ N * dvar
  dxmu2 = 2 * dsq * xmu
  dxmu = dxmu1 + dxmu2
  dmu = -1. * np.sum(dxmu,axis=0)
  dx2 = 1. / N * np.ones((N,D)) * dmu
  dx  = dxmu + dx2
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  #pass
  N,D = dout.shape
  xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
  dbeta = np.sum(dout,axis=0)
  dgammax = dout
  dgamma = np.sum(xhat * dgammax,axis=0)
  dxhat = gamma * dgammax
  dx = (1. - 1. / N) * np.ones((N,D)) * (np.sqrt(1. / N * np.sum(xmu ** 2,axis=0) + eps) - xmu * 1. / (2 * np.sqrt(1. / N * np.sum(xmu ** 2,axis=0) + eps))) / (1. / N * (np.sum(xmu ** 2,axis=0)) + eps) * dxhat
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    #pass
    #p = dropout_param.get('p')
    mask = (np.random.rand(*x.shape) >= p) / (1 - p)
    out = x * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    #pass
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    #pass
    dx = mask * dout
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  #pass
  F,CC,HH,WW = w.shape
  stride = conv_param.get('stride')
  pad = conv_param.get('pad')
  x_copy = x.copy()
  H_pre = x.shape[2]
  if pad:
    npad = ((0,0),(0,0),(pad, pad),(pad,pad))
    x_copy = np.pad(x,pad_width=npad,mode='constant',constant_values=0)
  #naive forward pass with loop
  #num_img = x_copy.shape[0]
  N,C,H,W = x_copy.shape
  if (H_pre + 2 * pad - HH) / stride != (H_pre + 2 * pad - HH) / float(stride):
      print "kernel not fit the img,assertion failed"
      return
  out_size = (H_pre + 2 * pad - HH) / stride + 1
  if C != CC:
      print "channel in img and filter not fit,assertion failed"
      return
  out = np.zeros((N,F,out_size,out_size))
  #print out.shape
  row,col = 0,0
  for index in range(N):
      for filt in range(F):
          while row + HH <= H:
              while col + WW <= W:
                  out[index,filt,row / stride,col / stride] = cal_inner_dot_bias_naive(x_copy,w,b,index,filt,row,col,HH,WW)
                  col += stride
              row += stride
	      col = 0
          row = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  #print out	
  return out, cache

def cal_inner_dot_bias_naive(x,w,b,index,filt,row,col,HH,WW):
    out = 0
    dot = 0.
    if HH != WW:
        print "row and col stride not match,assertion failed!"
        return
    for i in range(HH):
        for j in range(WW):
            dot = x[index,:,row + i,col + j].T.dot(w[filt,:,i,j])
            out += dot
    out += b[filt]
    return out
def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  #pass
  '''
  x,w,b,conv_param = cache
  stride = conv_param.get('stride')
  pad = conv_param.get('pad')
  F,CC,HH,WW = w.shape
  x_copy = x.copy()
  dw = np.zeros(w.shape)
  db = np.zeros(b.shape)
  #dw_mask = np.zeros_like(dw)
  if pad:
  	npad = ((0,0),(0,0),(pad, pad),(pad,pad))
  	x_copy = np.pad(x,pad_width=npad,mode='constant',constant_values=0)
  N,C,H,W = x_copy.shape
  dx = np.zeros(x_copy.shape)
  row,col = 0,0
  #rf_H = rf_W = HH
  for filt in range(F):
    while row + HH <= H:
        while col + WW <= W:
            dw[filt,:,:,:] += dout[:,filt,:,:] * np.sum(x_copy[:,:,],axis=0)
            #dx += w[filt,:,:,:]
            col += stride
        row += stride
        col = 0
    print dx.shape
    row = 0
  dw = 
  dx = dx[:,:,pad:-pad,pad:-pad]
  db = np.sum(dout,axis=(0,2,3))
  print dw
  '''
  x, w, b, conv_param = cache
  pad = conv_param['pad']
  stride = conv_param['stride']
  F, C, HH, WW = w.shape
  N, C, H, W = x.shape
  Hp = 1 + (H + 2 * pad - HH) / stride
  Wp = 1 + (W + 2 * pad - WW) / stride

  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  # Add padding around each 2D image (and respective gradient)
  # There may be a prettier way to do this but I can't think of any nice way at
  # least. You want to contribute to the boundary sums and in some cases the
  # only way to do that is by writing into the padding. I'm sure some very nasty
  # indexing trick will do; with lots of floors and ceils.
  padded = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
  padded_dx = np.pad(dx, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')

  for i in xrange(N): # ith example
    for j in xrange(F): # jth filter
      # Convolve this filter over windows
      for k in xrange(Hp):
        hs = k * stride
        for l in xrange(Wp):
          ws = l * stride

          # Window we applies the respective jth filter over (C, HH, WW)
          window = padded[i, :, hs:hs+HH, ws:ws+WW]

          # Compute gradient of out[i, j, k, l] = np.sum(window*w[j]) + b[j]
          db[j] += dout[i, j, k, l]
          dw[j] += window*dout[i, j, k, l]
          padded_dx[i, :, hs:hs+HH, ws:ws+WW] += w[j] * dout[i, j, k, l]

  # "Unpad"
  dx = padded_dx[:, :, pad:pad+H, pad:pad+W]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  #pass
  N,C,H,W = x.shape
  pool_height,pool_width,stride = pool_param.get('pool_height'),pool_param.get('pool_width'),pool_param.get('stride')
  out_hsize = (H - pool_height) / stride + 1
  out_wsize = (W - pool_width) / stride + 1
  out = np.zeros((N,C,out_hsize,out_wsize))
  for i in xrange(N):
    for j in xrange(C):
        for hs in xrange(out_hsize):
            for ws in xrange(out_wsize):
                hw = hs * stride
                ww = ws * stride
                window = x[i,j,hw:hw+pool_height,ww:ww+pool_width]
                out[i,j,hs,ws] = np.max(window)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  #pass
  x,pool_param = cache
  pool_height,pool_width,stride = pool_param.get('pool_height'),pool_param.get('pool_width'),pool_param.get('stride')
  dx = np.zeros_like(x)
  #dx_mask = np.zeros_like(dx)
  N,C,H,W = x.shape
  out_hsize = (H - pool_height) / stride + 1
  out_wsize = (W - pool_width) / stride + 1
  for i in xrange(N):
      for j in xrange(C):
          for hs in xrange(out_hsize):
              hw = hs * stride
              for ws in xrange(out_wsize):
                  ww = ws * stride
                  window = x[i,j,hw:hw+pool_height,ww:ww+pool_width]
                  m = np.max(window)
                  dx[i,j,hw:hw+pool_height,ww:ww+pool_width] = (window == m) * dout[i,j,hs,ws] 
                  #amax = np.unravel_index(window.argmax,)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
