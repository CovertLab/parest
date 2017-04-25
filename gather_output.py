
from __future__ import division

import os.path as pa

import sys

import numpy as np

FORCE = False

def load_pars(target_dir, n):
	dirs = [pa.join(target_dir, 'seed-{}'.format(i)) for i in xrange(n)]

	obj = np.column_stack([
		np.load(pa.join(d, 'obj.npy')) for d in dirs
		])

	pars = np.column_stack([
			np.load(pa.join(d, 'pars.npy')) for d in dirs
			])

	return obj, pars

target_dir = sys.argv[1]

obj_path = pa.join(target_dir, 'obj.npy')
pars_pars = pa.join(target_dir, 'pars.npy')

if not FORCE and pa.exists(obj_path) and pa.exists(pars_pars):
	print 'Skipping {}, output already exists'.format(target_dir)

else:
	print 'Gathering data for {}'.format(target_dir)

	(obj, pars) = load_pars(target_dir, 300)

	np.save(obj_path, obj)
	np.save(pars_pars, pars)
