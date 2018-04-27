
from __future__ import division

import os
import os.path

import argparse

import numpy as np

import problems
import optimize

parser = argparse.ArgumentParser()
parser.add_argument(
	'--seed', type = int, default = None
	)
parser.add_argument(
	'--n', type = int, default = 1
	)
parser.add_argument(
	'--force', action = 'store_true'
	)
parser.add_argument(
	'--naive', action = 'store_true'
	)

parser.add_argument(
	'--problem', type = str, default = 'all_scaled'
	)

args = parser.parse_args()

try:
	rules_and_weights = problems.DEFINITIONS[args.problem]

except KeyError:
	raise Exception('Unknown problem "{}".'.format(args.problem))

print 'Using problem definition "{}".'.format(args.problem)

for seed_offset in xrange(args.n):

	if args.seed is None:
		random_state = np.random.RandomState(None)
		print 'No seed provided; a random seed will be used.'
		print 'Output will not be saved.'
		outdir = None

	else:
		seed = args.seed + seed_offset

		random_state = np.random.RandomState(seed)
		print 'Using random seed {}.'.format(seed)

		subdir = 'naive' if args.naive else 'standard'

		outdir = os.path.join('out', 'history', 'new_target', subdir, 'seed-{}'.format(seed))

		print 'Output will be saved to {}.'.format(os.path.abspath(outdir))
		try:
			os.makedirs(outdir)

		except OSError:
			assert os.path.exists(outdir)

		if args.naive:
			print 'Using naive perturbations'

		else:
			print 'Using parsimonious perturbations'

		pars_path = os.path.join(outdir, 'pars.npy')
		obj_path = os.path.join(outdir, 'obj.npy')
		hist_path = os.path.join(outdir, 'hist.npy')

		hist = []
		def collect(epoch, iteration, constraint_penalty_weight, obj):
			hist.append((
				epoch,
				iteration,
				constraint_penalty_weight,
				obj.mass_eq + obj.energy_eq + obj.flux,
				obj.fit,
				))

	if outdir is None or args.force or not all(os.path.exists(p) for p in (pars_path, obj_path, hist_path)):

		callback = collect if (outdir is not None) else optimize.empty_callback

		(pars, obj) = optimize.estimate_parameters(
			rules_and_weights,
			random_state = random_state,
			naive = args.naive,
			random_direction = False,
			callback = callback
			)

		if outdir is not None:
			np.save(pars_path, pars)
			np.save(obj_path, np.array([
				obj.mass_eq,
				obj.energy_eq,
				obj.flux,
				obj.fit
				]))
			np.save(hist_path, np.array(hist, dtype = [
				('epoch', np.int64),
				('iter', np.int64),
				('const_pen', np.float64),
				('constraints', np.float64),
				('fit', np.float64),
				]))

			print 'Saved to {}'.format(outdir)

	else:
		print 'Skipped - output already exists'
