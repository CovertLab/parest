
from __future__ import division

from itertools import izip

import numpy as np
import matplotlib.pyplot as plt

import constants

MIN_FIT = 7.56725569871 # ideally this would be obtained programatically

from matplotlib.cm import RdBu
COLOR_OFFSET = 0.1

COLOR_NAIVE = (
	'crimson'
	# RdBu(0.0 + COLOR_OFFSET)
	)
COLOR_NEW = (
	'royalblue'
	# RdBu(1.0 - COLOR_OFFSET)
	)

from matplotlib import colors

COLOR_BOTH = tuple(
	(i + j)/2
	for (i, j) in izip(
		colors.hex2color(colors.get_named_colors_mapping()[COLOR_NAIVE]),
		colors.hex2color(colors.get_named_colors_mapping()[COLOR_NEW])
		)
	)

MARKER_STYLE = dict(
	markeredgewidth = 0.5,
	markeredgecolor = 'w',
	linestyle = 'none',
	)

MARKER_VALID = MARKER_STYLE.copy()
MARKER_VALID.update(
	marker = 'o',
	ms = 4,
	)

MARKER_INVALID = MARKER_STYLE.copy()
MARKER_INVALID.update(
	marker = 'X',
	ms = 5,
	)

SCALE = ( # TODO: decide if using the scaling is worth it
	#constants.RT * np.log(10)
	1.0
	)

obj_naive = np.load('out/history/naive/obj.npy')
valid_naive = np.load('out/history/naive/valid.npy')

obj_new = np.load('out/history/standard/obj.npy')
valid_new = np.load('out/history/standard/valid.npy')

plt.figure(figsize = (6, 6))

# plt.subplot(2, 2, 2)

f_naive = obj_naive[3, :] #- MIN_FIT
g_naive = obj_naive[:3, :].sum(0)

f_new = obj_new[3, :] #- MIN_FIT
g_new = obj_new[:3, :].sum(0)

plt.plot(f_naive[valid_naive]/SCALE, g_naive[valid_naive], c = COLOR_NAIVE, **MARKER_VALID)
plt.plot(f_naive[~valid_naive]/SCALE, g_naive[~valid_naive], c = COLOR_NAIVE, **MARKER_INVALID)

plt.plot(f_new[valid_new]/SCALE, g_new[valid_new], c = COLOR_NEW, **MARKER_VALID)
plt.plot(f_new[~valid_new]/SCALE, g_new[~valid_new], c = COLOR_NEW, **MARKER_INVALID)

plt.yscale('log')

plt.ylabel(r'$g(x)$')
plt.xlabel(r'$f(x)$')

plt.axvline(MIN_FIT/SCALE, lw = 0.5, ls = ':', c = 'k', zorder = -10)

plt.xlim(-10, 210)
plt.xticks([0, 50, 100, 150, 200])

plt.ylim(10**-17, 10**0)
plt.yticks(10.**np.arange(-16, 0, 2))

plt.savefig('figure2_reduced.pdf')
plt.savefig('figure2_reduced.png', dpi = 300)
