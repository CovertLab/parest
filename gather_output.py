
from __future__ import division

import os.path as pa

import numpy as np

import problems

FORCE = False

def load_pars(problem, n):
	dirs = [pa.join('out', problem, 'seed-{}'.format(i)) for i in xrange(n)]

	obj = np.column_stack([
		np.load(pa.join(d, 'obj.npy')) for d in dirs
		])

	pars = np.column_stack([
			np.load(pa.join(d, 'pars.npy')) for d in dirs
			])

	return obj, pars

for problem in problems.DEFINITIONS.viewkeys():
	obj_path = pa.join('out', problem, 'obj.npy')
	pars_pars = pa.join('out', problem, 'pars.npy')

	if not FORCE and pa.exists(obj_path) and pa.exists(pars_pars):
		print 'Skipping {}, files already exist'.format(problem)

	else:
		print 'Gathering data for {}'.format(problem)

		(obj, pars) = load_pars(problem, 300)

		np.save(obj_path, obj)
		np.save(pars_pars, pars)
