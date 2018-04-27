
from __future__ import division

from itertools import izip

import numpy as np
import matplotlib.pyplot as plt

import constants

MIN_FIT = 7.56725569871 # ideally this would be obtained programatically

from matplotlib.cm import RdBu
COLOR_OFFSET = 0.1

COLOR_NAIVE = (
	# 'crimson'
	# RdBu(0.0 + COLOR_OFFSET)
	np.array((94, 187, 71), np.float64)/255.
	)
COLOR_NEW = (
	# 'royalblue'
	# RdBu(1.0 - COLOR_OFFSET)
	np.array((225, 6, 133), np.float64)/255.
	)

from matplotlib import colors

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

plt.xlim(-10, 130)
plt.xticks([0, 20, 40, 60, 80, 100, 120])

plt.ylim(10**-16, 10**-2)
plt.yticks(10.**np.arange(-14, -2, 2))

plt.savefig('figure3.pdf')
# plt.savefig('figure3.png', dpi = 300)

print valid_naive.sum(), valid_naive.mean()
print valid_new.sum(), valid_new.mean()
