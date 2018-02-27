"""

This module is used to parse data from a comma-separated, table-like
format.  The format supports comments, multiple data types per file, and
default parameter assignments.  It collects the explicit data as well as
the metadata into a 'knowledge base' object.

# TODO: more documentation
# TODO: consider a format that distinguishes between keys and values
# TODO: investigate parser libraries
# TODO: tests

"""

from __future__ import division

from collections import OrderedDict, namedtuple
from itertools import izip

LINESEP = '\n'
FIELDSEP = ','
COMMENTCHAR = '#'
DATATYPECHAR = '@'
IDSEP = ':'
BRACKETS = ('(', ')')
ASSIGNCHAR = '='

ID = 'id'
FILEPATH = 'filepath'
LINENUM = 'linenum'
DATATYPE = 'datatype'

_RESERVED = (ID, FILEPATH, LINENUM, DATATYPE)

def idfunc_from_fields(*fieldnames):
	def idfunc(fieldvals):
		return ':'.join(str(fieldvals[fieldname]) for fieldname in fieldnames)

	return idfunc

_idfunc_default = idfunc_from_fields(DATATYPE, FILEPATH, LINENUM)

def escape_split(string, splitchars, remove = True):
	assert '{' not in string
	assert '}' not in string

	for i, char in enumerate(splitchars):
		string = string.replace('\\' + char, '{' + str(i) + '}')

	split = string.split(splitchars)
	out = []

	for substring in split:
		for i, char in enumerate(splitchars):
			if remove:
				replace = char

			else:
				replace = '\\' + char

			out.append(
				substring.replace('{' + str(i) + '}', replace)
				)

	return out

class KnowledgeBase(object):
	def __init__(self, names):
		self.datasets = {}

		for name in names:
			dataset = DataSet()

			self.datasets[name] = dataset
			setattr(self, name, dataset)

class DataSet(object):
	def __init__(self):
		self.entries = OrderedDict()

	def add_entry(self, entry):
		entry_id = getattr(entry, ID)

		assert entry_id not in self.entries.viewkeys()

		self.entries[entry_id] = entry

	def __getitem__(self, entry_id):
		return self.entries[entry_id]

	def __len__(self):
		return len(self.entries)

	def __iter__(self):
		return self.entries.itervalues()

	def __contains__(self, entry_id):
		return (entry_id in self.entries.viewkeys())

class ParsingError(Exception):
	pass

class Loader(object):
	def __init__(self):
		self.datatypes = {}

	def add_datatype(self, datatype):
		name = datatype.name

		assert name not in self.datatypes.viewkeys()

		self.datatypes[name] = datatype

	def load(self, filepaths):
		kb = KnowledgeBase(self.datatypes.keys())

		for path in filepaths:
			with open(path) as datafile:
				datatype = None
				unset_fields = None
				n_expected = None
				preset_fields = None

				for linenum, line in enumerate(datafile):
					cleaned = line.partition(COMMENTCHAR)[0].strip()

					if len(cleaned) == 0:
						pass

					elif cleaned.startswith(DATATYPECHAR):
						left, _, right = cleaned.lstrip(DATATYPECHAR).partition(BRACKETS[0])

						name = left.strip()

						try:
							datatype = self.datatypes[name]

						except KeyError:
							raise ParsingError('Unknown datatype "{}" on line {} in {}'.format(
								name, linenum, path
								))

						preset_fields = {}
						unset_fields = []

						preset_vals = right.rstrip(BRACKETS[1])

						if len(preset_vals) > 0:
							for preset_val in escape_split(preset_vals, FIELDSEP):
								left, _, right = preset_val.partition(ASSIGNCHAR)

								fieldname = left.strip()
								strval = right.strip()

								try:
									value = datatype.fields[fieldname].caster(strval)

								except KeyError:
									raise ParsingError('Unknown field name "{}" on line {} in {}'.format(
										fieldname, linenum, path
										))

								except ValueError:
									raise ParsingError('Failed to cast "{}" on line {} in {}'.format(
										strval, linenum, path
										))

								preset_fields[fieldname] = value

						for field in datatype.fields.viewvalues():
							if field.name not in preset_fields.viewkeys():
								unset_fields.append(field)

						n_expected = len(unset_fields)

					else:
						if datatype is None:
							raise ParsingError('No datatype or invalid datatype for line {} in {}'.format(
								linenum, path
								))

						if IDSEP in cleaned:
							left, _, right = cleaned.partition(IDSEP)
							idstr = left.strip()
							strvals = escape_split(right, FIELDSEP)

						else:
							idstr = None
							strvals = escape_split(cleaned, FIELDSEP)

						entry_dict = preset_fields.copy()
						entry_dict[FILEPATH] = path
						entry_dict[LINENUM] = linenum
						entry_dict[DATATYPE] = datatype.name

						n_actual = len(strvals)

						if n_actual != n_expected:
							raise ParsingError('Expected {} field(s) but found {} on line {} in {}'.format(
								n_expected, n_actual, linenum, path
								))

						for field, strval in izip(unset_fields, strvals):
							stripped = strval.strip()

							try:
								value = field.caster(stripped)

							except ValueError:
								raise ParsingError('Failed to cast "{}" on line {} in {}'.format(
									stripped, linenum, path
									))

							entry_dict[field.name] = value

						if idstr is None:
							idstr = datatype.idfunc(entry_dict)

						entry_dict[ID] = idstr

						kb.datasets[datatype.name].add_entry(
							datatype.entry_class(**entry_dict)
							)

		return kb

class DataType(object):
	def __init__(self, name, fields, idfunc = _idfunc_default):
		self.name = name

		self.fields = OrderedDict()
		for field in fields:
			self.fields[field.name] = field

		self.idfunc = idfunc

		fieldnames = self.fields.keys()
		fieldnames.extend(_RESERVED)

		self.entry_class = namedtuple(name, fieldnames)

class Field(object):
	def __init__(self, name, caster = str):
		assert name not in _RESERVED
		self.name = name
		self.caster = caster
