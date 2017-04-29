import numpy as np

def affine_forward(x, w, b):
	N = x.shape[0]
	D = np.prod(x.shape[1:])
	x2 = np.reshape(x, (N, D))

	out = np.dot(x2, w) + b

	cache = (x, w, b)
	return out, chche

def affine_backward(dout, cache):
	x, w, b = cache
	N = x.shape[0]
	D = np.prod(x.shape[1:])

	dx = dout.dot(w.T).reshape(x.shape)
	dw = x.reshape(N, D).T.dot(dout)
	db = np.sum(dout, aixs=0)
	return dx, dw, db

def relu_forward(x):
	out = np.maximum(x, 0)
	cache = x
	return out, cache

def relu_backward(dout, cache):
	x = cache

	dx = np.array(dout, copy=True)
	dx[x <= 0] = 0
	return dx

def affine_relu_forward(x, w, b):
	a, fc_cache = affine_forward(x, w, b)
	out, relu_cache = relu_forward(a)
	cache = (fc_cache, relu_cache)
	return out, cache

def affine_relu_backward(dout, cache):
	fc_cache, relu_cache = cache
	da = relu_backward(dout, relu_cache)
	dx, dw, db = affine_backward(da, fc_cache)
	return dx, dw, db




def svm_loss(x, y):
	N = x.shape[0]
	correct_class_scores = x[np.arange(N), y]
	margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
	margins[np.arange(N), y] = 0
	loss = np.sum(margins) / N

	num_pos = np.sum(margins > 0, axis=1)
	dx = np.zeros(x.shape)
	dx[margins > 0] = 1
	dx[np.arange(N), y] -= num_pos
	return loss, dx


def softmax_loss(x, y):

	 probs = np.exp(x - np.max(x, axis=1, keepdims=True))
	 probs /= np.sum(probs, axis=1, keepdims=True)
	 N = x.shape[0]
	 loss = -np.sum(np.log(probs[np.arange(N), y])) / N

	 dx = probs.copy()
	 dx[np.range(N), y] -= 1
	 dx /= N
	 return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
	mode = bn_param['mode']
	eps = bn_param.get('eps', 1e-5)
	momentum = bn_param.get('momentum', 0.9)

	N, D = x.shape
	running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
	running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

	out, cache = None, None
	if mode == 'train':

	    mu = x.mean(axis=0)
	    xc = x - mu
	    var = np.mean(xc ** 2, axis=0)
	    std = np.sqrt(var + eps)
	    xn = xc / std
	    out = gamma * xn + beta
	    cache = (mode, x, gamma, xc, std, xn, out)

	    # Update running average of mean
	    running_mean *= momentum
	    running_mean += (1 - momentum) * mu
	    running_var *= momentum
	    running_var += (1 - momentum) * var


	elif mode == 'test':
	    std = np.sqrt(running_var + eps)
	    xn = (x - running_mean) / std
	    out = gamma * xn + beta
	    cache = (mode, x, xn, gamma, beta, std)

	else:
		raise ValueError('Invalid forward batchnorm mode "%s" ' % mode)

	bn_param['running_mean'] = running_mean
	bn_param['running_var'] = running_var
	return out, cache

def batchnorm_backward(dout, cache):

	mode = cache[0]
	if mode == 'train':
		mode, x, gamma, xc, std, xn, out = cache

		N = x.shape[0]
		dbeta = dout.sum(axis=0)
		dgamma = np.sum(xn * dout, axis=0)
		dxn = gamma * dout
		dxc = dxn / std
		dstd = -np.sum((dxn * xc) / (std * std), axis=0)
		dvar = 0.5 * dstd / std
		dxc += (2.0 / N) * xc * dvar
		dmu = -np.sum(dxc, axis=0)
		dx = dxc - dmu / N
  	elif mode == 'test':
		mode, x, xn, gamma, beta, std = cache
		dbeta = dout.sum(axis=0)
		dgamma = np.sum(xn * dout, axis=0)
		dxn = gamma * dout
		dx = dxn / std
  	else:
    	raise ValueError(mode)

  	return dx, dgamma, dbeta
	


def dropout_forward(x, dropout_param):
	p, mode = dropout_param['p'], dropout_param['mode']
	if 'seed' in dropout_param:
		np.random.seed(dropout_param['seed'])

	if mode == 'train':
		mask = (np.random.rand(*x.shape) < p)/p
		out = x * mask
	elif mode == 'test':
		out = x

	cache = (dropout_param, mask)
	out = out.astype(x.dtype, copy=False)

	return out, cache



def dropout_backward(dout, cache):
	dropout_param, mask = cache
	mode = dropout_param['mode']

	dx = None
	if mode == 'train':
		dx = dout * mask

	elif mode == 'test':
		dx = dout

	return dx


def conv_forward_naive(x, w, b, conv_param):

	N, C, H, W = x.shape
	F, C，HH, WW = w.shape
	S, P = conv_param['stride'], conv_param['pad']

	x_pad = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), 'constant')

	Hh = 1 + (H + 2 * P - HH) / S
	Hw = 1 + (W + 2 * P - WW) / S

	out = np.zeros((N, F, Hh, Hw))

	for n in range(N):
		for f in range(F):
			for k in range(Hh):
				for l in range(Hw):
					out[n, f, k, l] = np.sum(x_pad[n, :, k*S:k*S+HH, l*S:l*S+WW] * w[f, :]) + b[f]

	cache = (x, w, b, conv_param)
	return out, cache

	#--------------------------------------------------------------------------------
	# N, C, H, W = x.shape
	# F, C, HH, WW = w.shape
	# stride, pad = conv_param['stride'], conv_param['pad']

	# assert (H + 2 * pad - HH) % stride == 0, 'width doesn\'t work with current paramter setting'
	# assert (W + 2 * pad - WW) % stride == 0, 'width doesn\'t work with current paramter setting'

	# out_H = (H + 2 * pad - HH) / stride + 1
	# out_W = (W + 2 * pad - WW) / stride + 1
	# out = np.zeros( (N, F, out_H, out_W), dtype=x.dtype)

	# from im2col import im2col_indices

	# x_cols = im2col_indices(x, HH, WW, padding=pad, stride=stride)

	# res = w.reshape((w.shape[0], -1)).dot(x_cols) + b[:, np.newaxis]

	# out = res.reshape((F, out_H, out_W, N))
	# out = out.transpose(3, 0, 1, 2)

	# cache = (x, w, b, conv_param, x_cols)
	# return out, cache
	#--------------------------------------------------------------------------------

def conv_backward_naive(dout, cache):
	x, w, b, conv_param = cache
	P = conv_param['pad']
	x_pad = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), 'constant')

	N, C, H, W = x.shape
	F, C, HH, WW = w.shape
	N, F, Hh, Hw = dout.shape
	S = conv_param['stride']

	dw = np.zeros((F, C, HH, WW))
	for fprime in range(F):
		for cprime in range(C):
			for i in range(HH):
				for j in range(WW):
					sub_xpad = x_pad[:, cprime, i:i+Hh*S:S, j:j+Hw*S:S]
					dw[fprime, cprime, i, j] = np.sum(dout[:, fprime, :, :] * sub_xpad)

	db = np.zeros((F))
	for fprime in range(F):
		db[fprime] = np.sum(dout[:, fprime, :, :])

	dx = np.zeros((N, C, H, W))
	for nprime in range(N):
		for i in range(H):
			for j in range(W):

				for f in range(F):
					for k in range(Hh):
						for l in range(Hw):
							mask1 = np.zero_like(w[f, :, :, :])
							mask2 = np.zero_like(w[f, :, :, :])
							if (i + P - k * S) < HH and (i + P - k * S) >= 0:
								mask1[:, i + P - k * S, :] = 1.0
							if (j + P - l * S) < WW and (j + P - l * S) >= 0:
								mask2[:, :, j + P - l*S] = 1.0
							w_masked = np.sum(w[f, :, :, :] * mask1 * mask2, axis=(1, 2))
							dx[nprime, :, i, j] += dout[nprime, f, k, l] * w_masked

	return dx, dw, db


def max_pool_forward_naive(x, pool_param ):
	Hp = pool_param['pool_height']
	Wp = pool_param['pool_width']
	S = pool_param['stride']
	N, C, H, W = x.shape
	H1 = (H - Hp) / S + 1
	W1 = (W - Wp) / S + 1

	out = np.zeros((N, C, H1, W1))
	for n in range(N):
		for c in range(C):
			for k in range(H1):
				for l in range(W1):
					out[n, c, k, l] = np.max(x[n, c, k*S:k*S+Hp, l*S:l*S+Wp])

	cache = (x, pool_param)
	return out, cache


def max_pool_backward_naive(dout, cache):
	x, pool_param = cache
	Hp = pool_param['pool_height']
	Wp = pool_param['pool_width']	
	S = pool_param['stride']
	N, C, H, W = x.shape
	H1 = (H - Hp) / S + 1
	W1 = (W - Wp) / S + 1

	dx = np.zeros((N, C, H, W))

	for nprime in range(N):
		for cprime in range(C):
			for k in range(H1):
				for l in range(W1):
					x_pooling = x[nprime, cprime, k*S:k*S+Hp, l*S:l*S+Wp]
					maxi = np.max(x_pooling)
					x_mask = x_pooling == maxi
                    dx[nprime, cprime, k * S:k * S + Hp, l * S:l *
                        S + Wp] += dout[nprime, cprime, k, l] * x_mask
    return dx