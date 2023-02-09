import numpy as np
import matplotlib.pyplot as plt
import os
import math


font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }


fig, ax1 = plt.subplots()
left, bottom, width, height = [0.15, 0.55, 0.20, 0.15]
ax2 = fig.add_axes([left, bottom, width, height])



ax1.set_xlabel('E', fontdict=font)
ax1.set_ylabel(r'$\sigma(\frac{2e^2}{h})$', fontdict=font)

with open('EEzvsTTz1_16_8_bk.dat') as f:
    lines = f.readlines()
    En1 = [float(line.split(';')[0]) for line in lines]
    Tn1 = [float(line.split(';')[1]) for line in lines]

ax1.plot(En1, Tn1, linewidth=0.8, label='W=8')
#legend = ax1.legend(loc='upper center', shadow=False, fontsize='x-large')


with open('EEzvsTTz1_16_10.dat') as f:
    lines = f.readlines()
    En2 = [float(line.split(';')[0]) for line in lines]
    Tn2 = [float(line.split(';')[1]) for line in lines]

ax1.plot(En2, Tn2, linewidth=0.8, label='W=10')
legend = ax1.legend(loc='upper left', shadow=False, fontsize='x-large')

with open('DISP01z1_16_8.dat') as f:
    lines = f.readlines()
    Dpx = [float(line.split(';')[0]) for line in lines]
    Dpy = [float(line.split(';')[1]) for line in lines]

ax2.scatter(Dpx, Dpy)
ax2.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',
    left='off',  # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off',
    labelleft='off')

plt.show()