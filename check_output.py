from __future__ import division

import os.path as pa

import numpy as np

import problems

n = 300

for problem in problems.DEFINITIONS:
	for d in (pa.join('out', problem, 'seed-{}'.format(i)) for i in xrange(n)):
		if not (pa.exists(pa.join(d, 'obj.npy')) & pa.exists(pa.join(d, 'pars.npy'))):
			print 'missing output for {}'.format(d)
