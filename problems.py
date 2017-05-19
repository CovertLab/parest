
from collections import OrderedDict

from data import kb
import structure

DEFAULT_RULES = (
	(lambda entry: entry.source == 'custom_saturation_limits', 0),
	)

DEFINITIONS = OrderedDict()

DEFINITIONS['data_agnostic'] = DEFAULT_RULES + ((lambda entry: True, 1.0),)

DEFINITIONS['no_data'] = DEFAULT_RULES + ((lambda entry: True, 0),)

def exclude_entry(excluded_entry):
	return (
		(lambda entry: entry.id == excluded_entry.id, 0),
		(lambda entry: True, 1)
		)

for entry in kb.concentration:
	if entry.compound in structure.compounds:
		problem_name = 'exclude_{}'.format(entry.id).replace(':', '_')

		DEFINITIONS[problem_name] = DEFAULT_RULES + exclude_entry(entry)

for entry in kb.standard_energy_of_formation:
	if entry.compound in structure.compounds:
		problem_name = 'exclude_{}'.format(entry.id).replace(':', '_')

		DEFINITIONS[problem_name] = DEFAULT_RULES + exclude_entry(entry)

import fitting

_KINETICS_TYPES = (
	'forward_catalytic_rate',
	'reverse_catalytic_rate',
	'substrate_saturation'
	)

DEFINITIONS['custom'] = DEFAULT_RULES + (
	(
		fitting.field_value_rule(
			datatype = ('concentration', 'standard_energy_of_formation'),
			compound = ('H2O', 'Pi', 'H')
			),
		1e1
		),
	(
		fitting.field_value_rule(
			datatype = ('concentration', 'relative_protein_count'),
			),
		1e0
		),
	(
		fitting.field_value_rule(
			datatype = _KINETICS_TYPES
			),
		1e-1
		),
	(
		fitting.field_value_rule(
			datatype = ('standard_energy_of_formation',),
			),
		1e-2
		),
	)

DEFINITIONS['custom2'] = DEFAULT_RULES + (
	(
		fitting.field_value_rule(
			datatype = ('concentration', 'standard_energy_of_formation'),
			compound = ('H2O', 'Pi', 'H')
			),
		1e1
		),
	(
		fitting.field_value_rule(
			datatype = _KINETICS_TYPES
			),
		1e0
		),
	(
		fitting.field_value_rule(
			datatype = ('concentration', 'relative_protein_count'),
			),
		1e-1
		),
	(
		fitting.field_value_rule(
			datatype = ('standard_energy_of_formation',),
			),
		1e-2
		),
	)

DEFINITIONS['no_kinetics'] = DEFAULT_RULES + (
	(
		fitting.field_value_rule(
			datatype = _KINETICS_TYPES,
			),
		0
		),
	(
		lambda entry: True,
		1e0
		)
	)

DEFINITIONS['no_proteomics'] = DEFAULT_RULES + (
	(
		fitting.field_value_rule(
			datatype = ('relative_protein_count'),
			),
		0
		),
	(
		lambda entry: True,
		1e0
		)
	)

DEFINITIONS['promote_fba_kinetics'] = DEFAULT_RULES + (
	(
		fitting.field_value_rule(
			datatype = _KINETICS_TYPES,
			reaction = 'FBA'
			),
		1e1
		),
	(
		lambda entry: True,
		1e0
		)
	)

DEFINITIONS['promote_fba_kinetics2'] = DEFAULT_RULES + (
	(
		fitting.field_value_rule(
			datatype = _KINETICS_TYPES,
			reaction = 'FBA'
			),
		1e2
		),
	(
		lambda entry: True,
		1e0
		)
	)

DEFINITIONS['upper_sat_limits'] = tuple()

def test():
	print '{} problem definitions:'.format(len(DEFINITIONS))
	print '\n'.join(DEFINITIONS.viewkeys())

	from fitting import build_fitting_tensors

	for name, rules_and_weights in DEFINITIONS.viewitems():
		build_fitting_tensors(*rules_and_weights)
		build_relative_fitting_tensor_sets(*rules_and_weights)

if __name__ == '__main__':
	test()
