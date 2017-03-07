
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
	if entry.compound in structure.active_compounds:
		problem_name = 'exclude_{}'.format(entry.id).replace(':', '_')

		DEFINITIONS[problem_name] = exclude_entry(entry)

for entry in kb.standard_energy_of_formation:
	if entry.compound in structure.active_compounds:
		problem_name = 'exclude_{}'.format(entry.id).replace(':', '_')

		DEFINITIONS[problem_name] = exclude_entry(entry)

# kinetics_sources = set()

# for entry in kb.forward_catalytic_rate:
# 	if entry.reaction in structure.active_reactions:
# 		kinetics_sources.add(entry.source)

# for entry in kb.reverse_catalytic_rate:
# 	if entry.reaction in structure.active_reactions:
# 		kinetics_sources.add(entry.source)

# for entry in kb.substrate_saturation:
# 	if entry.reaction in structure.active_reactions:
# 		kinetics_sources.add(entry.source)

# def exclude_source(source):
# 	return (
# 		(lambda entry: entry.source == source, 0),
# 		(lambda entry: True, 1)
# 		)

# for source in kinetics_sources:
# 	problem_name = 'exclude_kinetics_{}'.format(source)
# 	DEFINITIONS[problem_name] = exclude_source(source)

if __name__ == '__main__':
	print '{} problem definitions:'.format(len(DEFINITIONS))
	print '\n'.join(DEFINITIONS.viewkeys())

	from fitting import build_fitting_tensors

	for name, rules_and_weights in DEFINITIONS.viewitems():
		build_fitting_tensors(*rules_and_weights)
