
'''

The various parameter estimation problems are defined here, as well as
several convenience functions that can be used to build up new problems.  All
problems are stored in the global constant DEFINITIONS.

'''

from collections import OrderedDict

from data import kb
import structure
import fitting

_DEFAULT_EXCLUDE_DATATYPE = (
	# Pseudo-fitting data; not desired by default
	'upper_reactant_saturation_limit',
	'upper_product_saturation_limit',

	# SEOF data proved to be very inaccurate and was ultimately excluded
	'standard_energy_of_formation'
	)

_DEFAULT_RULES = (
	(
		fitting.field_value_rule(datatype = _DEFAULT_EXCLUDE_DATATYPE),
		0
		),
	)

def accept_all(value = 1e0):
	'''
	A rule-generation function that accepts everything.
	'''
	return (
		lambda entry: True,
		value
		)

DEFINITIONS = OrderedDict()

'''
The 'data agnostic' problem weights all data evenly.
'''

DEFINITIONS['data_agnostic'] = _DEFAULT_RULES + (accept_all(),)

'''
The 'no data' problem excludes all training data.
'''
DEFINITIONS['no_data'] = _DEFAULT_RULES + (accept_all(0),)

_KINETICS_TYPES = (
	'forward_catalytic_rate',
	'reverse_catalytic_rate',
	'substrate_saturation'
	)

from collections import defaultdict

def gather_by_fields(dataset, *fields):
	out = defaultdict(set)

	for entry in dataset:
		out[tuple(getattr(entry, field) for field in fields)].add(entry)

	return out

def gather_by_field(dataset, field):
	return {
		key[0]:value
		for key, value in gather_by_fields(dataset, field).viewitems()
		}

from itertools import izip

def normalize_by_number_of_observations(dataset, *fields):
	'''

	Creates a rule-weight pair that normalizes the weighting on the data by the
	number of observations.  This was found to give better fits, since data
	were often consistent and errors were usually consequent of a systemic
	change rather than experimental error.  E.g. chemical equilibria constants
	were found to not be totally consistent for the cell interior, despite
	having many (sometimes 10+) observations on each constant.

	'''
	return tuple(
		(
			fitting.field_value_rule(**{
				field:(value,)
				for (field, value) in izip(fields, values)
				}),
			1.0/len(entries)
			)
		for (values, entries) in gather_by_fields(
			dataset,
			*fields
			).viewitems()
		)

'''
The 'all scaled' ruleset is the default.
'''

DEFINITIONS['all_scaled'] = (
	_DEFAULT_RULES
	+ normalize_by_number_of_observations(kb.concentration, 'datatype', 'compound')
	+ normalize_by_number_of_observations(kb.equilibrium, 'datatype', 'reaction')
	+ normalize_by_number_of_observations(kb.forward_catalytic_rate, 'datatype', 'reaction')
	+ normalize_by_number_of_observations(kb.reverse_catalytic_rate, 'datatype', 'reaction')
	+ normalize_by_number_of_observations(kb.reactant_saturation, 'datatype', 'reaction', 'compound')
	+ normalize_by_number_of_observations(kb.product_saturation, 'datatype', 'reaction', 'compound')
	) + (
	(
		fitting.field_value_rule( # relative data hard to scale by # of obs; instead scaling by total # of data sets (3)
			datatype = ('relative_protein_count',),
			),
		1.0/3 # currently using three sets of proteomics data
		),
	)

'''
Variants on the 'all scaled' data set that include the saturation penalty, with
increasing weight.
'''

DEFINITIONS['all_scaled_upper_sat_limits_1e-1'] = (
	(
		fitting.field_value_rule(source = ('custom_saturation_limits',)),
		1e-1
		),
	) + DEFINITIONS['all_scaled']

DEFINITIONS['all_scaled_upper_sat_limits_1e0'] = (
	(
		fitting.field_value_rule(source = ('custom_saturation_limits',)),
		1e0
		),
	) + DEFINITIONS['all_scaled']

DEFINITIONS['all_scaled_upper_sat_limits_1e1'] = (
	(
		fitting.field_value_rule(source = ('custom_saturation_limits',)),
		1e1
		),
	) + DEFINITIONS['all_scaled']

DEFINITIONS['all_scaled_upper_sat_limits_1e2'] = (
	(
		fitting.field_value_rule(source = ('custom_saturation_limits',)),
		1e2
		),
	) + DEFINITIONS['all_scaled']

if __name__ == '__main__':
	test()
