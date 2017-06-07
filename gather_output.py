
from __future__ import division

import os.path as pa

import sys

import numpy as np

FORCE = False

def load_pars(target_dir, n):
	obj = []
	pars = []

	errors = []

	for d in [pa.join(target_dir, 'seed-{}'.format(i)) for i in xrange(n)]:
		for filename, collection in (
				('obj.npy', obj),
				('pars.npy', pars)
				):
			path = pa.join(d, filename)

			try:
				a = np.load(path)

			except IOError:
				errors.append(path)

			else:
				collection.append(a)

	if errors:
		raise Exception('Failed to open files:\n'+'\n'.join(errors))

	obj = np.column_stack(obj)
	pars = np.column_stack(pars)

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
