
import sys

from itertools import izip

ALIGNMENT = {
	'left':'',
	'right':'>',
	'center':'^'
	}

class Field(object):
	def __init__(self, name, format_character, width = None, alignment = 'right', oversize_fill_char = '#'):
		self.name = name

		if width is None:
			width = len(self.name)

		self.alignment = ALIGNMENT[alignment]
		self.width = width

		self.format_character = format_character

		self.oversize_fill = oversize_fill_char*width

	def format(self, value):
		out = '{:{alignment}{width}{format}}'.format(
			value,
			alignment = self.alignment,
			width = self.width,
			format = self.format_character
			)

		if len(out) > self.width:
			out = self.oversize_fill

		return out

	def format_name(self):
		return '{:{alignment}{width}.{width}}'.format(
			self.name,
			alignment = self.alignment,
			width = self.width,
			)

class Table(object):
	def __init__(self, fields, margin = ' ', separator = '  ', reprint_header_every = 40, out = sys.stdout, flush = True):
		self.fields = fields
		self.margin = margin
		self.separator = separator
		self.reprint_header_every = reprint_header_every
		self.out = out
		self.flush = flush

		self.time_since_header = None

		formatted = []
		for field in self.fields:
			formatted.append(
				field.format_name()
				)

		self.lines = self.margin + self.separator.join('-' * len(f) for f in formatted) + self.margin

		self.header = self.margin + self.separator.join(formatted) + self.margin

	def write(self, *values):
		if self.time_since_header is None or self.time_since_header > self.reprint_header_every:
			self.out.write(self.lines + '\n')
			self.out.write(self.header + '\n')
			self.out.write(self.lines + '\n')

			self.time_since_header = 0

		formatted = []
		for field, value in izip(self.fields, values):
			formatted.append(
				field.format(value)
				)

		self.out.write(self.margin + self.separator.join(formatted) + self.margin + '\n')

		self.time_since_header += 1

		if self.flush:
			self.out.flush()

if __name__ == '__main__':
	fields = [
		Field('Iteration', 'n'),
		Field('Words', 's'),
		Field('Longer words', 's'),
		Field('Float', '.2f', 7),
		Field('Exp', '.2e', 10),
		]

	words = ['these', 'are', 'words', 'of', 'varying', 'lengths']

	dt = 0.17

	table = Table(fields)

	for i in xrange(1000):
		table.write(
			i,
			words[i%len(words)],
			words[i%len(words)],
			i * dt,
			i * dt,
			)
