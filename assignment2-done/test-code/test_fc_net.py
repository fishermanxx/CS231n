import numpy as np
import test_layer.py

class TwoLayerNet(object):

	def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
				 weight_scale=1e-3, reg=0.0):

		self.params = {}
		self.reg = reg
		self.D = input_dim
		self.M = hidden_dim
		self.C = num_classes

		w1 = weight_scale * np.random.randn(self.D, self.M)
		b1 = np.zeros(self.M)
		w2 = weight_scale * np.random.randn(self.M, self.C)
		b2 = np.zeros(self.C)

		self.params.updates({
				'W1': w1,
				'b1': b1,
				'W2': w2,
				'b2': b2
			})

	def loss(self, X, y=None):

		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']

		X = X.reshape(X.shape[0], self.D)
		hidden_layer, cache_hidden_layer = affine_relu_forward(X, W1, b1)
		scores, cache_scores = affine_forward(hidden_layer, W2, b2)

		if y is None:
			return scores

		loss, grads = 0, {}
		
		data_loss, dscores = softmax_loss(scores, y)
		reg_loss = 0.5 * self.reg * np.sum(W1**2)
		reg_loss += 0.5 * self.reg * np.sum(W2**2)
		loss = data_loss + reg_loss

		dx2, dW2, db2 = affine_backward(dscores, cache_scores)
		dW2 += self.reg * W2

		dx1, dW1, db1 = affine_relu_backward(dx2, cache_hidden_layer)
		dW1 += self.reg * W1

		grads.update({
				'W1': dW1,
				'b1': db1,
				'W2': dW2,
				'b2': db2
			})

		return loss, grads


class FullyConnectedNet(object):

	def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
				 dropout=0, use_batchnorm=False, reg=0.0,
				 weight_scale=1e-2, dtype=np.float32, seed=None):
		self.use_batchnorm = use_batchnorm
		self.use_dropout = dropout > 0
		self.reg = reg
		self.num_layers = 1 + len(hidden_dims)
		self.dtype = dtype
		self.params = {}

		if type(hidden_dims) != list:
			raise ValueError('hidden_dim has to be a list')

		self.L = len(hidden_dims) + 1
		self.N = input_dim
		self.C = num_classes
		dims = [self.N] + hidden_dims + [self.C]
		
		Ws = {
			'W' + str(i+1): weight_scale * np.random.randn(dims[i], dims[i+1]) for i in range(len(dims) - 1)
		}

		b = {
			'b' + str(i+1): np.zeros(dims[i+1]) for i in range(len(dims) - 1)
		}

		self.params.update(Ws)
		self.params.update(b)

		for k, v in self.params.iteritems():
			self.params[k] = v.astype(dtype)

	def loss(self, X, y=None):

		X = X.astype(self.dtype)
		mode = 'test' if y is None else 'train'

		hidden = {}
		hidden['h0'] = X.reshape(X.shape[0], np.prod(X.shape[1:]))

		for i in range(self.L):
			idx = i + 1

			w = self.params['W' + str(idx)]
			b = self.params['b' + str(idx)]
			h = hidden['h' + str(idx - 1)]

			if idx == self.L:
				h, cache_h = affine_forward(h, w, b)
				hidden['h' + str(idx)] = h
				hidden['cache_h' + str(idx)] = cache_h
			else:
				h, cache_h = affine_relu_forward(h, w, b)
				hidden['h' + str(idx)] = h
				hidden['cache_h' + str(idx)] = cache_h

		scores = hidden['h' + str(self.L)]

		if mode == 'test':
			return scores

		loss, grads = 0.0, {}

		data_loss, dscores = softmax_loss(scores, y)
		
		reg_loss = 0
		for w in [self.params[f] for f in self.params.keys() if f[0] == 'W']:
			reg_loss += 0.5 * self.reg * np.sum(w ** 2)

		loss = data_loss + reg_loss

		hidden['dh' + str(self.L)] = dscores
		for i in range(self.L)[::-1]:
			idx = i + 1
			dh = hidden['dh' + str(idx)]
			h_cache = hidden['cache_h' + str(idx)]
			
			if idx == self.L:
				dh, dw, db = affine_backward(dh, h_cache)
				hidden['dh' + str(idx - 1)] = dh
				hidden['dW' + str(idx)] = dw
				hidden['db' + str(idx)] = db
			else:
				dh, dw, db = affine_relu_backward(dh, h_cache)
				hidden[]
				hidden['dh' + str(idx - 1)] = dh
				hidden['dW' + str(idx)] = dw
				hidden['db' + str(idx)] = db


		list_dw = {key[1:]: val + self.reg * self.params[key[1:]]
					for key, val in hidden.iteritems() if key[:2] == 'dW'}
		list_db = {key[1:]: val
					for key, val in hidden.iteritems() if key[:2] == 'db'}

		grads = {}
		grads.update(list_dw)
		grads.update(list_db)

		return loss, grads







