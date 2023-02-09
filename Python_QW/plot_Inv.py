import numpy as np
import matplotlib.pyplot as plt
import os
import math


font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

plt.figure(1)
plt.subplot(121)




#fig, ax1 , ax2 = plt.subplots(1,2)
#left, bottom, width, height = [0.2, 0.6, 0.2, 0.1]
#ax2 = fig.add_axes([left, bottom, width, height])



plt.ylabel('E', fontdict=font)
plt.xlabel(r'$\sigma$', fontdict=font)

with open('EEzvsTTz1_20.dat') as f:
    lines = f.readlines()
    En1 = [float(line.split(';')[0])*100 for line in lines]
    Tn1 = [float(line.split(';')[1]) for line in lines]

#ax1.plot(Tn1, En1, linewidth=0.4, label='W=10')
plt.ylim(ymin=0, ymax=4)
plt.xlim(xmin=0, xmax=4)
plt.plot(Tn1, En1, linewidth=0.8)


plt.subplot(122)
plt.xlabel(r'$k$', fontdict=font)

with open('EEzvsTTz1_20.dat') as f:
    lines = f.readlines()
    En2 = [float(line.split(';')[0]) for line in lines]
    Tn2 = [float(line.split(';')[1]) for line in lines]

aux = [int(x+1) for x in Tn2]
aux2 = [xe*100 for xe in En2]

x1 = -1
x2 = []
xe = []
i=0
for x in aux:
    if x != x1:
        x2.append(x)
        if i == 0:
            xe.append(0.32)
        else:
            xe.append(aux2[i])
        x1 = x
    i = i+1

print(x2)
print(xe)

xe2 = []

plt.ylim(ymin=0, ymax=4)
plt.xlim(xmin=-1, xmax=1)


plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',
    left='off',  # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off',
    labelleft='off')

plt.axvline(0, color='black', linestyle='dashed', linewidth = 0.2)
for i in np.linspace(-1,1,50):
    xe2.append(i**2 + xe[0])

plt.axhline(xe[0], color='black', linestyle='dashed', linewidth = 0.2)
plt.plot(np.linspace(-0.5,0.5,50), xe2, linewidth=0.8, color='C0')
xe2.clear()

for i in np.linspace(-1,1,50):
    xe2.append(i**2 + xe[1])

plt.axhline(xe[1], color='black', linestyle='dashed', linewidth = 0.2)
plt.plot(np.linspace(-0.5,0.5,50), xe2, linewidth=0.8, color='C0')
xe2.clear()

for i in np.linspace(-1,1,50):
    xe2.append(i**2 + xe[2])

plt.axhline(xe[2], color='black', linestyle='dashed', linewidth = 0.2)
plt.plot(np.linspace(-0.5,0.5,50), xe2, linewidth=0.8, color='C0')
xe2.clear()

for i in np.linspace(-1,1,50):
    xe2.append(i**2 + xe[3])

plt.axhline(xe[3], color='black', linestyle='dashed', linewidth = 0.2)
plt.plot(np.linspace(-0.5,0.5,50), xe2, linewidth=0.8, color='C0')
xe2.clear()

plt.show()