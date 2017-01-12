
import numpy as np

def unbounded(x, upperbounds, lowerbounds):
	return (x > upperbounds) | (x < lowerbounds)

def unbounded_to_random(x, upperbounds, lowerbounds):
	ub = unbounded(x, upperbounds, lowerbounds)

	x = x.copy()

	x[ub] = (
		lowerbounds
		+ np.random.random(size = x.shape) * (upperbounds - lowerbounds)
		)[ub]

	return x
