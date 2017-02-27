
import numpy as np

def unbounded(x, upperbounds, lowerbounds):
	return (x > upperbounds) | (x < lowerbounds)

# this is a good candidate for cythonization
def unbounded_to_random(x, upperbounds, lowerbounds, random_state = np.random):
	ub = unbounded(x, upperbounds, lowerbounds)

	if ub.any():
		x = x.copy()

		x[ub] = (
			lowerbounds
			+ random_state.random_sample(size = x.shape) * (upperbounds - lowerbounds)
			)[ub]

	return x
