import numpy as np

def sgd(w, dw, config=None):
	
	if config is None: config = {}
	config.etdefault('learning_rate', 1e-2)

	w -= config['learning_rate'] * dw
	return w, config

def sgd_momentum(w, dw, config=None):
	
	if config is None: config = {}
	config.setdefault('learning_rate', 1e-2)
	config.setdefault('momentum', 0.9)
	v = config.get('velocity', np.zero_like(w))

	next_w = None
	next_v = config['momentum'] * v - config['learning_rate'] * dw
	next_w = w + next_v

	config['velocity'] = next_v
	return next_w, config


def rmsprop(x, dx, config=None):

    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

	config['cache'] = config['cache'] * config['decay_rate'] +\
		(1 - config['decay_rate']) * dx**2
	next_x = x - config['learning_rate'] * dx / (np.sqrt(config['cache']
	                                                      + config['epsilon']))
	return next_x, config


def adam(x, dx, config=None):

	if config is None: config = {}
	config.setdefault('learning_rate', 1e-3)
	config.setdefault('beta1', 0.9)
	config.setdefault('beta2', 0.999)
	config.setdefault('epsilon', 1e-8)
	config.setdefault('m', np.zeros_like(x))
	config.setdefault('v', np.zeros_like(x))
	config.setdefault('t', 0)


	learning_rate = config['learning_rate']
	beta1 = config['beta1']
	beta2 = config['beta2']
	epsilon = config['epsilon']

	config['t'] += 1
	config['m'] = beta1 * config['m'] + (1 - beta1) * dx
	config['v'] = beta2 * config['v'] + (1 - beta2) * dx**2
	mt_hat = config['m'] / (1 - (beta1)**config['t'])
	vt_hat = config['v'] / (1 - (beta2)**config['t'])
	next_x = x - learning_rate * mt_hat / (np.sqrt(vt_hat + epsilon))


	return next_x, config
