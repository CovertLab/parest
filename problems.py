
from collections import OrderedDict

from data import kb
import structure

DEFINITIONS = OrderedDict()

DEFINITIONS['data_agnostic'] = tuple()

DEFINITIONS['no_data'] = ((lambda entry: True, 0),)

def exclude_entry(excluded_entry):
	return (
		(lambda entry: entry.id == excluded_entry.id, 0),
		(lambda entry: True, 1)
		)

for entry in kb.concentration:
	if entry.compound in structure.compounds:
		problem_name = 'exclude_{}'.format(entry.id).replace(':', '_')

		DEFINITIONS[problem_name] = exclude_entry(entry)

for entry in kb.standard_energy_of_formation:
	if entry.compound in structure.compounds:
		problem_name = 'exclude_{}'.format(entry.id).replace(':', '_')

		DEFINITIONS[problem_name] = exclude_entry(entry)

def test():
	print '{} problem definitions:'.format(len(DEFINITIONS))
	print '\n'.join(DEFINITIONS.viewkeys())

	from fitting import build_fitting_tensors

	for name, rules_and_weights in DEFINITIONS.viewitems():
		build_fitting_tensors(*rules_and_weights)
		build_relative_fitting_tensor_sets(*rules_and_weights)

if __name__ == '__main__':
	test()
