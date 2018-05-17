
from collections import OrderedDict

from data import kb
import structure
import fitting

_DEFAULT_EXCLUDE_DATATYPE = (
	'upper_reactant_saturation_limit',
	'upper_product_saturation_limit',
	'standard_energy_of_formation'
	)

_DEFAULT_RULES = (
	(
		fitting.field_value_rule(datatype = _DEFAULT_EXCLUDE_DATATYPE),
		0
		),
	)

def accept_all(value = 1e0):
	return (
		lambda entry: True,
		value
		)

def exclude_entry(excluded_entry):
	return (
		(lambda entry: entry.id == excluded_entry.id, 0),
		accept_all()
		)

DEFINITIONS = OrderedDict()

DEFINITIONS['data_agnostic'] = _DEFAULT_RULES + (accept_all(),)

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

def scale_weights(rules_and_weights, scale):
	return tuple( (rule, weight * scale) for rule, weight in rules_and_weights)

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

DEFINITIONS['all_scaled_Pi_H2O_1e1'] = (
	(
		fitting.field_value_rule(datatype = 'concentration', compound = 'Pi'),
		1e1
		),
	(
		fitting.field_value_rule(datatype = 'concentration', compound = 'H2O'),
		1e1
		),
	) + DEFINITIONS['all_scaled']

def test():
	print '{} problem definitions:'.format(len(DEFINITIONS))
	print '\n'.join(DEFINITIONS.viewkeys())

	import fitting

	for name, rules_and_weights in DEFINITIONS.viewitems():
		fitting.build_fitting_tensors(*rules_and_weights)
		fitting.build_relative_fitting_tensor_sets(*rules_and_weights)

if __name__ == '__main__':
	test()
