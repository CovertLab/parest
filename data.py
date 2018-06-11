
'''

Defines the datatypes that will be loaded, and loads them into a 'knowledge
base' object (kb).

'''

from __future__ import division

import os

import loader

DIR = os.path.join(os.path.dirname(__file__), 'data')
EXT = '.dat'

_loader = loader.Loader()

_loader.add_datatype(loader.DataType(
	'description',
	(loader.Field('text'),)
	))

_loader.add_datatype(loader.DataType(
	'source',
	(loader.Field('name'),)
	))

_loader.add_datatype(loader.DataType(
	'compound',
	(
		loader.Field('name'),
		loader.Field('source')
		),
	loader.idfunc_from_fields(loader.DATATYPE, 'name')
	))

_loader.add_datatype(loader.DataType(
	'reaction',
	(
		loader.Field('name'),
		loader.Field('source')
		),
	loader.idfunc_from_fields(loader.DATATYPE, 'name')
	))

_loader.add_datatype(loader.DataType(
	'reactant',
	(
		loader.Field('reaction'),
		loader.Field('compound'),
		loader.Field('stoichiometry', int),
		loader.Field('source')
		),
	loader.idfunc_from_fields(loader.DATATYPE, 'reaction', 'compound')
	))

_loader.add_datatype(loader.DataType(
	'product',
	(
		loader.Field('reaction'),
		loader.Field('compound'),
		loader.Field('stoichiometry', int),
		loader.Field('source')
		),
	loader.idfunc_from_fields(loader.DATATYPE, 'reaction', 'compound')
	))

_loader.add_datatype(loader.DataType(
	'molar_mass',
	(
		loader.Field('compound'),
		loader.Field('molar_mass', float),
		loader.Field('source')
		),
	loader.idfunc_from_fields(loader.DATATYPE, 'compound', 'source')
	))

_loader.add_datatype(loader.DataType(
	'standard_energy_of_formation',
	(
		loader.Field('compound'),
		loader.Field('standard_energy_of_formation', float),
		loader.Field('source')
		),
	loader.idfunc_from_fields(loader.DATATYPE, 'compound', 'source')
	))

_loader.add_datatype(loader.DataType(
	'concentration',
	(
		loader.Field('compound'),
		loader.Field('concentration', float),
		loader.Field('source')
		),
	loader.idfunc_from_fields(loader.DATATYPE, 'compound', 'source')
	))

_loader.add_datatype(loader.DataType(
	'forward_catalytic_rate',
	(
		loader.Field('reaction'),
		loader.Field('k_cat', float),
		loader.Field('source')
		),
	loader.idfunc_from_fields(loader.DATATYPE, 'reaction', 'source')
	))

_loader.add_datatype(loader.DataType(
	'reverse_catalytic_rate',
	(
		loader.Field('reaction'),
		loader.Field('k_cat', float),
		loader.Field('source')
		),
	loader.idfunc_from_fields(loader.DATATYPE, 'reaction', 'source')
	))

_loader.add_datatype(loader.DataType(
	'reactant_saturation',
	(
		loader.Field('reaction'),
		loader.Field('compound'),
		loader.Field('index', int),
		loader.Field('K_M', float),
		loader.Field('source')
		),
	loader.idfunc_from_fields(loader.DATATYPE, 'reaction', 'compound', 'source')
	))

_loader.add_datatype(loader.DataType(
	'product_saturation',
	(
		loader.Field('reaction'),
		loader.Field('compound'),
		loader.Field('index', int),
		loader.Field('K_M', float),
		loader.Field('source')
		),
	loader.idfunc_from_fields(loader.DATATYPE, 'reaction', 'compound', 'source')
	))

_loader.add_datatype(loader.DataType(
	'relative_flux',
	(
		loader.Field('reaction'),
		loader.Field('flux', float),
		loader.Field('source')
		),
	loader.idfunc_from_fields(loader.DATATYPE, 'reaction', 'source')
	))

_loader.add_datatype(loader.DataType(
	'relative_protein_count',
	(
		loader.Field('reaction'),
		loader.Field('count', float),
		loader.Field('source')
		),
	loader.idfunc_from_fields(loader.DATATYPE, 'reaction', 'source')
	))

_loader.add_datatype(loader.DataType(
	'upper_reactant_saturation_limit',
	(
		loader.Field('reaction'),
		loader.Field('compound'),
		loader.Field('index', int),
		loader.Field('ratio', float),
		loader.Field('source')
		),
	loader.idfunc_from_fields(loader.DATATYPE, 'reaction', 'compound', 'index', 'source')
	))

_loader.add_datatype(loader.DataType(
	'upper_product_saturation_limit',
	(
		loader.Field('reaction'),
		loader.Field('compound'),
		loader.Field('index', int),
		loader.Field('ratio', float),
		loader.Field('source')
		),
	loader.idfunc_from_fields(loader.DATATYPE, 'reaction', 'compound', 'index', 'source')
	))

_loader.add_datatype(loader.DataType(
	'equilibrium',
	(
		loader.Field('reaction'),
		loader.Field('equilibrium_constant', float),
		loader.Field('source')
		),
	loader.idfunc_from_fields(loader.DATATYPE, 'reaction', 'source')
	))

_paths = list(
	os.path.join(d[0], f)
	for d in os.walk(DIR)
	for f in d[2]
	if f.endswith(EXT)
	)

kb = _loader.load(_paths)
