
from collections import OrderedDict

from data import kb
import structure
import fitting

_DEFAULT_EXCLUDE = dict(
	datatype = (
		'custom_saturation_limits',
		'standard_energy_of_formation'
		)
	)

_DEFAULT_RULES = (
	(
		fitting.field_value_rule(**_DEFAULT_EXCLUDE),
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

for entry in kb.concentration:
	if entry.compound in structure.compounds:
		problem_name = 'exclude_{}'.format(entry.id).replace(':', '_')

		DEFINITIONS[problem_name] = _DEFAULT_RULES + exclude_entry(entry)

for entry in kb.standard_energy_of_formation:
	if entry.compound in structure.compounds:
		problem_name = 'exclude_{}'.format(entry.id).replace(':', '_')

		DEFINITIONS[problem_name] = _DEFAULT_RULES + exclude_entry(entry)

_KINETICS_TYPES = (
	'forward_catalytic_rate',
	'reverse_catalytic_rate',
	'substrate_saturation'
	)

DEFINITIONS['custom'] = _DEFAULT_RULES + (
	(
		fitting.field_value_rule(
			datatype = ('concentration', 'standard_energy_of_formation'),
			compound = ('H2O', 'Pi', 'H', 'ATP', 'AMP', 'ADP')
			),
		1e1
		),
	(
		fitting.field_value_rule(datatype = ('standard_energy_of_formation',)),
		1e-1
		),
	accept_all()
	)

DEFINITIONS['custom2'] = _DEFAULT_RULES + (
	(
		fitting.field_value_rule(
			datatype = ('concentration', 'standard_energy_of_formation'),
			compound = ('H2O', 'Pi')
			),
		1e1
		),
	accept_all()
	)

DEFINITIONS['no_seof'] = _DEFAULT_RULES + (
	(
		fitting.field_value_rule(datatype = ('standard_energy_of_formation',)),
		0
		),
	accept_all()
	)

DEFINITIONS['seof_1e-1'] = _DEFAULT_RULES + (
	(
		fitting.field_value_rule(datatype = ('standard_energy_of_formation',)),
		1e-1
		),
	accept_all()
	)

DEFINITIONS['no_seof_PEP_conc_100x'] = _DEFAULT_RULES + (
	(
		fitting.field_value_rule(datatype = ('standard_energy_of_formation',)),
		0
		),
	(
		fitting.field_value_rule(
			datatype = ('concentration',),
			compound = ('PEP',)
			),
		1e2
		),
	accept_all()
	)

DEFINITIONS['no_seof_gap_kinetics_100x'] = _DEFAULT_RULES + (
	(
		fitting.field_value_rule(datatype = ('standard_energy_of_formation',)),
		0
		),
	(
		fitting.field_value_rule(
			datatype = _KINETICS_TYPES,
			reaction = 'GAP'
			),
		1e2
		),
	accept_all()
	)

DEFINITIONS['no_seof_gap_kinetics_100x_scaled'] = _DEFAULT_RULES + (
	(
		fitting.field_value_rule(datatype = ('standard_energy_of_formation',)),
		0
		),
	(
		fitting.field_value_rule(
			datatype = _KINETICS_TYPES,
			reaction = 'GAP'
			),
		1e1
		),
	accept_all(1e-1)
	)

DEFINITIONS['upper_sat_limits'] = tuple()

DEFINITIONS['upper_sat_limits_x10'] = (
	(lambda entry: entry.source == 'custom_saturation_limits', 10.0),
	(lambda entry: True, 1.0),
	)

DEFINITIONS['Keq_e-1'] = _DEFAULT_RULES + (
	(
		lambda entry: entry.datatype == 'equilibrium',
		1e-1
		),
	accept_all()
	)

DEFINITIONS['Keq_half'] = _DEFAULT_RULES + (
	(
		lambda entry: entry.datatype == 'equilibrium',
		1e-1
		),
	accept_all()
	)

DEFINITIONS['Keq_FBA_TPI_e-1'] = _DEFAULT_RULES + (
	(
		fitting.field_value_rule(
			datatype = ('equilibrium'),
			reaction = ('FBA', 'TPI', 'FBA_TPI')
			),
		1e-1
		),
	accept_all()
	)

DEFINITIONS['Keq_FBA_TPI_ENO_e-1'] = _DEFAULT_RULES + (
	(
		fitting.field_value_rule(
			datatype = ('equilibrium'),
			reaction = ('FBA', 'TPI', 'FBA_TPI', 'ENO')
			),
		1e-1
		),
	accept_all()
	)

def test():
	print '{} problem definitions:'.format(len(DEFINITIONS))
	print '\n'.join(DEFINITIONS.viewkeys())

	from fitting import build_fitting_tensors

	for name, rules_and_weights in DEFINITIONS.viewitems():
		build_fitting_tensors(*rules_and_weights)
		build_relative_fitting_tensor_sets(*rules_and_weights)

if __name__ == '__main__':
	test()
