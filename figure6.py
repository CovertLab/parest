
from __future__ import division

import os

import numpy as np

import matplotlib.pyplot as plt

import structure
import constants

from figure5 import make_clean_directory

def main(input_directory, output_directory):
	pars = np.load(os.path.join(input_directory, 'pars.npy'))[
		:, np.load(os.path.join(input_directory, 'valid.npy'))
		]

	specific_activity_data = [ # in micromol/min/mg
		('PGI', 212),
		('PGK', 98),
		('PGK', 480),
		('GPM', 124),
		('ENO', 158),
		('ENO', 260),
		('PYK', 124),
		('PPS', 8.9)
		]

	molecular_weight_data = { # in kDa per molecule
		'PGI':62,
		'PGK':42,
		'GPM':27,
		'ENO':90,
		'PYK':51,
		'PPS':150
		}

	# this could be simplified by noting 1 kDa/molecule = 1 kg/mol

	Nav = 6.022e23

	da_per_gram = Nav
	kda_per_mg = da_per_gram/1000/1000

	micromol_per_mol = 1e6

	s_per_min = 60

	kcat_f_values = np.array([
		constants.RT * np.log(
			value * molecular_weight_data[rxn] * Nav / kda_per_mg / micromol_per_mol / s_per_min
			/ constants.K_STAR
			)
		for (rxn, value) in specific_activity_data
		])

	kcat_f_matrix = np.row_stack([
		-structure.kcat_f_matrix[structure.reactions.index(rxn), :]
		for (rxn, value) in specific_activity_data
		])

	predicted = kcat_f_matrix.dot(pars)

	residuals = kcat_f_values[:, None] - predicted

	indexing = np.array([
		structure.reactions.index(rxn)
		for (rxn, value) in specific_activity_data
		])

	sorting = np.argsort(indexing)

	residuals = residuals[sorting, :]
	indexing = indexing[sorting]

	DPI = 1000

	import utils.residuals

	unique = np.unique(indexing)

	fig = utils.residuals.plot(residuals, indexing)

	make_clean_directory(output_directory)

	fig.savefig(os.path.join(output_directory, 'specific_activity.pdf'), dpi = DPI)

	with open(os.path.join(output_directory, 'key.txt'), 'w') as f:
		for unique_index in unique:
			f.write(':'.join([
				structure.reactions[unique_index]
				])+'\n')

if __name__ == '__main__':
	main('out/all_scaled', 'figure6')
