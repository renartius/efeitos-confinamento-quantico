#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
#import repy as re
import os
import math

L = 4
W = 8 #4,6
N = 18

LxLy = 18

Fano2 = 1

loop = 6000
#energia = 0.000008
energia = 0.0000081

NDz1 = N
temp_NWz1 = int((NDz1 - W)/2)
temp_DOT = int((NDz1 - LxLy)/2)

NWz1 = NDz1 - 2 * temp_NWz1
MCz1 = 4
MQPC1z1 = L
MQPC2z1 = L
MOQDz1 = LxLy #int(math.sqrt(A))
NMz1 = 2*MCz1+MQPC1z1+MOQDz1+MQPC2z1

Fano = NDz1
#Fano = temp_NWz1 + NWz1

Dpx = []
Dpy = []
En = []
Tn = []
# PARAMETER(NDz1=11, MCz1=10, MQPC1z1=3, NMz1=2 * MCz1 + MQPC1z1)
# PARAMETER(NDz1=55, MCz1=10, MQPC1z1=55, MQPC2z1=55, MOQDz1=55, NMz1=2 * MCz1 + MQPC1z1 + MOQDz1 + MQPC2z1, NWz1=33)

# PARAMETER(NDz1=21, MCz1=10, MQPC1z1=20, MQPC2z1=10, MOQDz1=21, NMz1=2 * MCz1 + MQPC1z1 + MOQDz1 + MQPC2z1, NWz1=3)
#

# Dispositivo
'''
p1x = np.linspace(1,MCz1,MCz1)
p1y = np.linspace(1,NDz1,NDz1)
p2x = np.linspace(MCz1+1,MCz1+MQPC1z1,MQPC1z1)
iy = (NDz1-NWz1)/2
p2y = np.linspace(iy,iy-1+NWz1,NWz1)
p3x = np.linspace(MCz1+MQPC1z1+1,MCz1+MQPC1z1+MCz1,MCz1)
p3y = p1y
p1xv,p1yv = np.meshgrid(p1x,p1y)
p2xv,p2yv = np.meshgrid(p2x,p2y)
p3xv,p3yv = np.meshgrid(p3x,p3y)
plt.plot(p1xv,p1yv,marker='.',linestyle='none',color='black')
plt.plot(p2xv,p2yv,marker='.',linestyle='none',color='black')
plt.plot(p3xv,p3yv,marker='.',linestyle='none',color='black')

plt.show()
'''

DISPOz1 = np.zeros((NMz1, NDz1), dtype='float', order='F')
GFMEz1 = np.zeros((NMz1, NDz1), dtype='complex', order='F')
HCz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
HQPC1z1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
HQPC2z1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
HOQDz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
HGz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
QW1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')

MDUXz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
MDUXB1z1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
MDUXB2z1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
CLz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
UUz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
GUz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
CLXz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
CRXz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
VHXz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
VHXB1z1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
VHXB2z1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
GTz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
GRz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
GIz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
XXz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
XX1z1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
XXAz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
XX1Az1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
HIz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
XIz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
GTvz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')
GRvz1 = np.zeros((NDz1, NDz1), dtype='complex', order='F')

HICz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
HIXz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
XIXz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
GIICz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
GIXz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')

file210 = open("EEzvsTTz1_"+ str(LxLy)+"_"+str(NWz1) +".dat", "w")
file340 = open("DISP01z1_"+ str(LxLy)+"_"+str(NWz1) +".dat", "w")
file360 = open("GvsFMEz1_"+ str(LxLy)+"_"+str(NWz1) +".dat", "w")

zrz1 = complex(1.0, 0.0)
ziz1 = complex(0.0, 1.0)
z0z1 = complex(0.0, 0.0)
zriz1 = zrz1 + ziz1


# DEFINICAO DA HETEROESTRUTURA:
# PARAMETRO DE REDE: "azz" (Amgstrons)
azz1 = 20.0

# #############FORTRAN ###########################

# LarguraL = (ni - 1) * azz1
# Largura medio da parte imaginaria daenergia
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
sz1 = 0.0000001

# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# PARAMETROS TIGHT BINDING USADOS(fundo da banda de conducao do GaAs)
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# Energia do sitio(GaAs): EA; Energia do sitio(AlGaAs): EB
# Energia hopping(GaAs): VA; Energia hopping(AlGaAs): VB
# #Energia hopping GaAs / AlGaAs: V
VAz1 = abs(-56.86608298 / (azz1 * azz1))
VBz1 = abs(-56.86608298 / (azz1 * azz1))
# VBz1 = -41.45840653 / (azy1 * azz1)
EIz1 = 1000 * VAz1
EAz1 = 2 * VAz1 * 2
EBz1 = 2 * VBz1 * 2 + 0.1
VHz1 = abs(-np.sqrt((VAz1 * VBz1)))
print("%f %f",(EIz1,EAz1))

# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# HC: hamiltoniano tipo tight binding da cadeia transversal do contato.
# HC(i, j): elementos de matriz do hamiltoniano.i: columna, j: fila.
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
for ixz1 in range(0, NDz1):
    for jxz1 in range(0, NDz1):
        HCz1[jxz1, ixz1] = z0z1
#@ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
for imz1 in range(0, NDz1):
    HCz1[imz1, imz1] = EAz1 * zrz1
    if imz1+1 < NDz1:
        HCz1[imz1+1, imz1] = -VAz1 * zrz1
        HCz1[imz1, imz1 + 1] = -VAz1 * zrz1

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
'''
#WBz1 = NWz1 - 1
#yyz1 = 1
#do iyz1 = 1, WBz1
#yy1z1 = yyz1 + iyz1
#do imz1 = 1, 1
#HCz1((yy1z1) + (imz1 - 1) * NDz1, (yy1z1) + (imz1 - 1) * NDz1) = EAz1 * zrz1
#end do
#end do
'''
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @

# @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @

# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** *DISPOSITIVO MESOSCOPICO ** ** ** ** ** ** ** ** ** ** ** ** ** *
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$      1) DIODO RESONANTE 2 D     $$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$        2) QPC        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$      3) T - STUB     $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$        4)DOUBLE BEND        $$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$           5) OQD           $$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
for ixz1 in range(0,NDz1):
    for jxz1 in range(0, NDz1):
        HQPC1z1[jxz1, ixz1] = z0z1


for ixz1 in range(0,NDz1):
    for jxz1 in range(0,NDz1):
        HQPC2z1[jxz1, ixz1] = z0z1


for ixz1 in range(0,NDz1):
    for jxz1 in range(0, NDz1):
        HOQDz1[jxz1, ixz1] = z0z1

for ixz1 in range(0,NDz1):
    for jxz1 in range(0, NDz1):
        HICz1[jxz1, ixz1] = z0z1

# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
for ixz1 in range(0,NDz1):
    for jxz1 in range(0, NDz1):
        HGz1[jxz1, ixz1] = z0z1

for imz1 in range(0, NDz1):
    HGz1[imz1, imz1] = EIz1 * zrz1
    if imz1+1 < NDz1:
        HGz1[imz1+1, imz1] = -VAz1 * zrz1
        HGz1[imz1, imz1 + 1] = -VAz1 * zrz1


# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** contato ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
HCz1 = HGz1.copy()

WBz1 = Fano
yyz1 = 0
for iyz1 in range(1, WBz1+1):
    yy1z1 = yyz1 + iyz1
    for imz1 in range(1, 2):
        HCz1[yy1z1 + (imz1 - 1) * NDz1-1, yy1z1 + (imz1 - 1) * NDz1-1] = EAz1 * zrz1

# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** QPC1 ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
HQPC1z1 = HGz1.copy()

WBz1 = NWz1
yyz1 = temp_NWz1 #4
for iyz1 in range(1, WBz1+1):
    yy1z1 = yyz1 + iyz1
    for imz1 in range(1, 2):
        HQPC1z1[yy1z1 + ((imz1 - 1) * NDz1)-1, yy1z1 + ((imz1 - 1) * NDz1)-1] = EAz1 * zrz1

print(HGz1[0, 0])
print(HQPC1z1[0, 0])
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** QPC2 ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
HQPC2z1 = HQPC1z1.copy()

# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** OQD ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
HOQDz1 = HGz1.copy()


WBz1 = Fano-Fano2
yyz1 = 0
for iyz1 in range(1, WBz1+1-(2*temp_DOT)):
    yy1z1 = yyz1 + iyz1
    for imz1 in range(1, 2):
        HOQDz1[yy1z1 + (imz1 - 1) * NDz1-1, yy1z1 + (imz1 - 1) * NDz1-1] = EAz1 * zrz1

# @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @

# @ @ @ @ @ @ @ @ @ @ @ @ @IMPUREZA CONTROLAVEL @ @ @ @ @ @ @ @ @ @ @ @ @ @
HICz1 = HGz1.copy()

WBz1 = NDz1
yyz1 = 0
for iyz1 in range(1, WBz1+1):
    yy1z1 = yyz1 + iyz1
    for imz1 in range(1, 2):
        HICz1[yy1z1 + (imz1 - 1) * NDz1-1, yy1z1 + (imz1 - 1) * NDz1-1] = EAz1 * zrz1

# ** ** ** ** ** ** ** ** *IMPUREZAS ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

# ** ** ** ** ** ** ** ** ** CADEIA 1 ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** CENTRO ** ** ** ** ** ** ** ** ** **
'''
#NIz1 = 11
#do iyz1 = 1, NIz1
#HICz1(5 + iyz1, 5 + iyz1) = (EIz1) * zrz1
#end do
#@ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @

##yyz1 = 2
##do iyaz1 = 1, 5
##   wpz1 = 7
##   do iyz1 = 1, wpz1
##   yy1z1 = yyz1 + iyz1
##   do imz1 = 1, 1
## HICz1((yy1z1) + (imz1 - 1) * NDz1, (yy1z1) + (imz1 - 1) * NDz1) = EAz1 * zrz1
# #HICz1((yy1z1) + (imz1 - 1) * NDz1, (yy1z1) + (imz1 - 1) * NDz1) = (EAz1 + 0.142) * zrz1
##   end do
##   end do
##   yyz1 = yyz1 + 11
##end do
# @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
'''
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

for iaz1 in range(0, NDz1):
    for jz1 in range(0, NDz1):
        MDUXz1[jz1, iaz1] = z0z1

for ibz1 in range(0, NDz1):
    MDUXz1[ibz1, ibz1] = zrz1

# @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
for iaz1 in range(0, NDz1):
    for jz1 in range(0, NDz1):
        MDUXB1z1[jz1, iaz1] = z0z1

for iaz1 in range(0, NDz1):
    for jz1 in range(0, NDz1):
        MDUXB2z1[jz1, iaz1] = z0z1

# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
piz1 = 0.
for iyz1 in range(0, 1):

    for ixz1 in range(0, loop):
        vv1z1 = 1.
        Ev1z1 = 2.0 * VAz1 - 2.0 * VAz1 * np.cos(piz1 * vv1z1 / (NDz1 + 1))
        piz1 = 3.141592654
        # Ez1 = 0.0 + 1e-20 + ixz1 * ((4) * VAz1 + Ev1z1 - 0.00001) / 90000
        # #Ez1 = 0.01339
        Ez1 = 0.0 + 1e-20 + ixz1 * energia

        ff1 = iyz1
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        #  ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        #  ** ** ** ** ** ** ** *CONTATOS SEMI - INFINITOS: CL, CR(CL=CR) ** ** ** ** ** ** ** ** **
        #  ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        #  ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        for inz1 in range(1, NDz1+1):
            vvz1 = 0
            for jnz1 in range(1, NDz1+1):
                vvz1 = 1 + vvz1
                CC1z1 = np.sqrt((2.0 / (NDz1+1)))
                piz1 = 3.141592654
                UUz1[jnz1-1, inz1-1] = CC1z1 * np.sin(piz1 * vvz1 * inz1 / (NDz1 + 1))

        for ivz1 in range(1, NDz1+1):
            Evz1 = 2.0 * VAz1 - 2.0 * VAz1 * np.cos(piz1 * ivz1 / (NDz1 + 1))
            if (Ez1 >= 0) and (Ez1 < Evz1):
                CAz1 = np.cos(np.arccos((Ez1 - Evz1) / (2. * VAz1) + 1.))*zrz1 - np.sin(np.arccos((Ez1 - Evz1) / (2. * VAz1) + 1.))*ziz1
            else:
                CAz1 = np.cos(np.arccos((Ez1 - Evz1) / (2. * VAz1) - 1.))*zrz1 - np.sin(np.arccos((Ez1 - Evz1) / (2. * VAz1) - 1.))*ziz1
            GUz1[ivz1-1, ivz1-1] = (1. / VAz1) * CAz1

        #CLz1 = re.mult_matrix_NxN2(UUz1, re.mult_matrix_NxN(GUz1, np.transpose(UUz1)))
        CLz1 = np.matmul(UUz1, np.matmul(GUz1, np.transpose(UUz1)))
        # print(np.matmul(UUz1, np.matmul(GUz1, np.transpose(UUz1))))


# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
#       METODO RECURSIVO PARA CALCULAR GT = Gm'm(n'n) e GR = Gmm(n'n)
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** *CAMPO B EM Z, GAUGE DE LANDAU A = (Ax, Ay, Az) = (-By, 0, 0) = (-Ban, 0, 0) ** ** ** *
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# CALCULAR A MATRIZ MDUXz SIMILAR QUE A CALCULADA GUz(funcao de Green contato)
        fluxo1 = ff1 / 4000000.
        # fluxo1 = 0.0003
        for ivz1 in range(1, NDz1+1):
            BA1z1 = np.cos(2 * piz1 * ivz1 * fluxo1) * zrz1 + np.sin(2 * piz1 * ivz1 * fluxo1) * ziz1
            BA2z1 = np.cos(2 * piz1 * ivz1 * fluxo1) * zrz1 - np.sin(2 * piz1 * ivz1 * fluxo1) * ziz1
            MDUXB1z1[ivz1-1, ivz1-1] = BA1z1
            MDUXB2z1[ivz1-1, ivz1-1] = BA2z1

        VHXB1z1 = -VAz1 * MDUXB1z1
        VHXB2z1 = -VAz1 * MDUXB2z1

        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        #  ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        VHXz1 = -VAz1 * MDUXz1
        # ** ** ** ** ** ** ** ** **
        CLXz1 = CLz1.copy()
        CRXz1 = CLz1.copy()
        # ** ** ** ** ** ** ** ** **
        GTz1 = CRXz1.copy()
        GRz1 = CRXz1.copy()

        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        #  ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        #  ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        #  ** ** ** ** ** ** *CONTATO1 ** ** ** ** ** ** ** *
        #  ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        HIz1 = HCz1.copy()
        XIz1 = 1E+30 * (Ez1 * MDUXz1 - HIz1 + sz1 * MDUXz1 * ziz1)
        XIz1 = np.linalg.inv(XIz1)#re.invxz1(XIz1, NDz1)
        GIz1 = 1E+30 * XIz1
        QW1 = GIz1.copy()

        jxz1 = 0

        for imz1 in range(MCz1, 0, -1):
            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
            #  ** ** ** ** ** ** ** ** DISPOSITIVO ** ** ** ** ** ** ** ** **
            jyz1 = 0
            jxz1 = jxz1 + 1
            for nxz1 in range(1, NDz1+1):
                jyz1 = jyz1 + 1
                DISPOz1[jxz1-1, nxz1-1] = np.real(HIz1[jyz1-1, jyz1-1])

            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
            VHXz1 = -VAz1 * MDUXz1

            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
            XXAz1 = 1E+30 * (MDUXz1 - np.matmul(GIz1, np.matmul(VHXz1, np.matmul(GRz1, VHXz1))))
            XXAz1 = np.linalg.inv(XXAz1)#re.invxz1(XXAz1, NDz1)
            XXz1 = 1E+30 * XXAz1
            GTz1 = np.matmul(GTz1, np.matmul(VHXz1, np.matmul(XXz1, GIz1)))
            GRz1 = np.matmul(XXz1, GIz1)


        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        # ** ** ** ** ** ** QPC1 ** ** ** ** **
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        HIz1 = HQPC1z1.copy()
        XIz1 = 1E+30 * (Ez1 * MDUXz1 - HIz1 + sz1 * MDUXz1 * ziz1)
        XIz1 = np.linalg.inv(XIz1)#re.invxz1(XIz1, NDz1)
        GIz1 = 1E+30 * XIz1
        # GIz1 = QW1

        jxz1 = MCz1
        for imz1 in range(MQPC1z1, 0, -1):
            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
            # ** ** ** ** ** ** ** *DISPOSITIVO ** ** ** ** ** ** ** ** ** **
            jyz1 = 0
            jxz1 = jxz1 + 1
            for nxz1 in range(1, NDz1+1):
                jyz1 = jyz1 + 1
                DISPOz1[jxz1-1, nxz1-1] = np.real(HIz1[jyz1-1, jyz1-1])

            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
            XXAz1 = 1E+30 * (MDUXz1 - np.matmul(GIz1, np.matmul(VHXz1, np.matmul(GRz1, VHXz1))))
            XXAz1 = np.linalg.inv(XXAz1)#re.invxz1(XXAz1, NDz1)
            XXz1 = 1E+30 * XXAz1
            GTz1 = np.matmul(GTz1, np.matmul(VHXz1, np.matmul(XXz1, GIz1)))
            GRz1 = np.matmul(XXz1, GIz1)

        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        # ** ** ** ** ** ** OQD ** ** ** ** **
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        HIz1 = HOQDz1.copy()
        XIz1 = 1E+30 * (Ez1 * MDUXz1 - HIz1 + sz1 * MDUXz1 * ziz1)
        XIz1 = np.linalg.inv(XIz1)#re.invxz1(XIz1, NDz1)
        GIz1 = 1E+30 * XIz1
        # GIz1 = QW1

        # ** ** ** ** *CADEIA 1 ** ** ** ** **
        HIXz1 = HICz1.copy()
        XIXz1 = 1E+30 * (Ez1 * MDUXz1 - HIXz1 + sz1 * MDUXz1 * ziz1)
        XIXz1 = np.linalg.inv(XIXz1)#re.invxz1(XIXz1, NDz1)
        GIICz1 = 1E+30 * XIXz1
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        # ** ** ** ** *CADEIA 2 ** ** ** ** **
        #  ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

        # CBz1 = (MOQDz1 + 1) / 2
        # jxbz1 = -(MOQDz1 + 1) / 2

        jxz1 = MCz1 + MQPC1z1
        #  ###########################################################
        # REDE DE ANTIDOTS
        #  ##########################################################

        PR1 = 11
        # #do imz1 = MOQDz1 / PR1, 1, -1

        # #do im1z1 = 1, PR1
        #  ###########################################################
        for imz1 in range(MOQDz1, 0, -1):
            '''
            # ###########################################################
            # #BILLAR QUANTICO
            # jxbz1 = jxbz1 + 1
            # HOQDz1 = HGz1
            # ysenz1 = NWz1 + NAYz1 / 2. * (1.+dcos(piz1 * jxbz1 / CBz1))
            # WY1 = NINT(yseny1)


            # yyz1 = 0
            # do iyz1 = 1, Wz1
            # yy1z1 = yyz1 + iyz1
            # do imxz1 = 1, 1
            # HOQDz1((yy1z1) + (imxz1 - 1) * NDz1, (yy1z1) + (imxz1 - 1) * NDz1) = EAz1 * zrz1
            # end do
            #end do

            #HIz1 = HOQDz1
            #XIz1 = 1E+30 * (Ez1 * MDUXz1 - HIz1 + ssz1 * MDUXz1 * ziz1)
            #re.invxz1(XIz1, NDz1)
            #GIz1 = 1E+30 * XIz1
            #  ###########################################################
            #  ###########################################################

            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
            # ** ** ** ** ** ** *IMPUREZAS CONTROLAVEIS ** ** ** ** ** ** ** **
            #imz1 es variavel
            #im1z1 = 1
            # @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @CADEIA DO CENTRO @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
            ##if ((im1z1 >= 3) and (im1z1.LE.9)) :
            ##HIz1=HICz1
            ##GIXz1=GIICz1
            ## else
            ##HIz1=HOQDz1
            ##GIXz1=GIz1
            ##end if
            # @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
            # @ @ @ @ @ @ @ @ @ @ @ @ @ @ CADEIAS DA ESQUERDA E DIREITA @ @ @ @ @ @ @ @ @ @ @ @ @ @
            # if (imz1.EQ.7) :
            #GIXz1=GIICz1
            # else
            # if (imz1.EQ.1) :
            #GIXz1=GIICz1
            # else
            #GIXz1=GIz1
            #end if
            #end if
            '''
            # @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
            # com impureza trocar GI por GIX (sem impureza GI)
            # @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
            # ** ** ** ** ** ** ** ** * DISPOSITIVO ** ** ** ** ** ** ** ** ** *
            jyz1 = 0
            jxz1 = jxz1 + 1
            for nxz1 in range(1, NDz1 + 1):
                jyz1 = jyz1 + 1
                DISPOz1[jxz1 - 1, nxz1 - 1] = np.real(HIz1[jyz1 - 1, jyz1 - 1])

            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
            #  ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
            XXAz1 = 1E+30 * (MDUXz1-np.matmul(GIz1, np.matmul(VHXB2z1, np.matmul(GRz1, VHXB1z1))))
            XXAz1 = np.linalg.inv(XXAz1)#re.invxz1(XXAz1, NDz1)
            XXz1 = 1E+30 * XXAz1
            GTz1 = np.matmul(GTz1, np.matmul(VHXB1z1, np.matmul(XXz1, GIz1)))
            GRz1 = np.matmul(XXz1, GIz1)

            # #end do
            #  #######################################################

            #  #######################################################
            # #end do

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        # ** ** ** ** ** ** QPC2 ** ** ** ** **
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        HIz1 = HQPC2z1.copy()
        XIz1 = 1E+30 * (Ez1 * MDUXz1-HIz1+sz1 * MDUXz1 * ziz1)
        XIz1 = np.linalg.inv(XIz1)#re.invxz1(XIz1, NDz1)
        GIz1 = 1E+30 * XIz1
        # GIz1 = QW1

        jxz1 = MCz1+MQPC1z1+MOQDz1
        for imz1 in range(MQPC2z1, 0, -1):
            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
            # ** ** ** ** ** ** ** ** * DISPOSITIVO ** ** ** ** ** ** ** ** **
            jyz1 = 0
            jxz1 = jxz1+1
            for nxz1 in range(1, NDz1+1):
                jyz1 = jyz1+1
                DISPOz1[jxz1-1, nxz1-1] = np.real(HIz1[jyz1-1, jyz1-1])

            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
            XXAz1 = 1E+30 * (MDUXz1-np.matmul(GIz1, np.matmul(VHXz1, np.matmul(GRz1, VHXz1))))
            XXAz1 = np.linalg.inv(XXAz1)#re.invxz1(XXAz1, NDz1)
            XXz1 = 1E+30 * XXAz1
            GTz1 = np.matmul(GTz1, np.matmul(VHXz1, np.matmul(XXz1, GIz1)))
            GRz1 = np.matmul(XXz1, GIz1)

        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        # ** ** ** ** ** ** * CONTATO2 ** ** ** ** ** ** ** *
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        HIz1 = HCz1.copy()
        XIz1 = 1E+30 * (Ez1 * MDUXz1-HIz1+sz1 * MDUXz1 * ziz1)
        # re.invxz1(XIz1, NDz1)
        # GIz1=1E+30 * XIz1
        GIz1 = QW1.copy()

        # jxz1=MCz1+MQPC1z1
        jxz1 = MCz1+MQPC1z1+MOQDz1+MQPC2z1
        for imz1 in range(MCz1, 0, -1):
            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
            # ** ** ** ** ** ** ** ** * DISPOSITIVO ** ** ** ** ** ** ** ** **
            jyz1 = 0
            jxz1 = jxz1+1
            for nxz1 in range(1, NDz1+1):
                jyz1 = jyz1+1
                DISPOz1[jxz1-1, nxz1-1] = np.real(HIz1[jyz1-1, jyz1-1])

            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
            # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
            XXAz1 = 1E+30 * (MDUXz1-np.matmul(GIz1, np.matmul(VHXz1, np.matmul(GRz1, VHXz1))))
            XXAz1 = np.linalg.inv(XXAz1)#re.invxz1(XXAz1, NDz1)
            XXz1 = 1E+30 * XXAz1
            GTz1 = np.matmul(GTz1, np.matmul(VHXz1, np.matmul(XXz1, GIz1)))
            GRz1 = np.matmul(XXz1, GIz1)

        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        XX1Az1 = 1E+30 * (MDUXz1-np.matmul(CLXz1, np.matmul(VHXz1, np.matmul(GRz1, VHXz1))))
        XX1Az1 = np.linalg.inv(XX1Az1)#re.invxz1(XX1Az1, NDz1)
        XX1z1 = 1E+30 * XX1Az1
        GTz1 = np.matmul(GTz1, np.matmul(VHXz1, np.matmul(XX1z1, CLXz1)))
        GRz1 = np.matmul(XX1z1, CLXz1)

        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        # ** ** ** ** ** ** ** ** ** * ARQUIVO DO DISPOSITIVO ** ** ** ** ** ** ** ** ** **
        for mxz1 in range(1, NMz1+1):
            for nxz1 in range(1, NDz1+1):
                if DISPOz1[mxz1-1, nxz1-1] == EAz1:
                    if ixz1 == 0:
                        #print("xy %i %i: %f and %f  " % (mxz1-1, nxz1-1,DISPOz1[mxz1-1, nxz1-1],EAz1))
                        Dpx.append((mxz1-1) * azz1)
                        Dpy.append((nxz1 - 1) * azz1)
                        file340.write(str((mxz1-1) * azz1)+";"+str((nxz1-1) * azz1)+"\n")

        file340.flush()
        os.fsync(file340)
        # file340.close()
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        #           PROBABILIDADE DE TRANSMISSAO(TT) E REFLEXAO(RR)
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        #              C0EFICIENTES DA MATRIZ S: t(u, v), r(u, v)
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        GTvz1 = np.matmul(np.transpose(UUz1), np.matmul(GTz1, UUz1))
        GRvz1 = np.matmul(np.transpose(UUz1), np.matmul(GRz1, UUz1))

        TTuvz1 = 0.
        RRuvz1 = 0.
        TTz1 = 0.
        for IKz1 in range(1, NDz1+1):
            vz1 = IKz1

            for IK1z1 in range(1, NDz1+1):
                uz1 = IK1z1
                Evz1 = (2. * VAz1) - 2. * VAz1 * np.cos((piz1 * vz1) / ((NDz1 + 1)*1.))
                Euz1 = (2. * VAz1) - 2. * VAz1 * np.cos((piz1 * uz1) / ((NDz1 + 1)*1.))
                if 0 <= Ez1 < Evz1:
                    if not math.isnan((Ez1 - Evz1) / (2. * VAz1) + 1.):
                        CF1z1 = math.sin(math.acos((Ez1 - Evz1) / (2. * VAz1) + 1.))
                    else:
                        CF1z1 = 1E+30

                else:
                    if not math.isnan((Ez1 - Evz1) / (2. * VAz1) - 1.):
                        CF1z1 = np.sin(np.arccos((Ez1 - Evz1) / (2. * VAz1) - 1.))
                    else:
                        CF1z1 = 1E+30

                if 0 <= Ez1 < Euz1:
                    if not math.isnan((Ez1 - Euz1) / (2. * VAz1) + 1.):
                        CF2z1 = np.sin(np.arccos((Ez1-Euz1) / (2. * VAz1)+1.))
                    else:
                        CF2z1 = 1E+30
                else:
                    if not math.isnan((Ez1 - Euz1) / (2. * VAz1) - 1.):
                        CF2z1 = np.sin(np.arccos((Ez1-Euz1) / (2. * VAz1)-1.))
                    else:
                        CF2z1 = 1E+30

                tuvz1 = 2. * VAz1 * np.sqrt(np.abs(CF2z1 * CF1z1)) * ziz1 * (GTvz1[uz1-1, vz1-1])
                #Tz1 = tuvz1.real**2 + tuvz1.imag*(tuvz1.imag*(-1)) + tuvz1.real*tuvz1.imag + tuvz1.real*(tuvz1.imag*(-1))
                #TTuvz1 = TTuvz1 + Tz1
                Tz1 = tuvz1 *tuvz1.conjugate()
                TTuvz1 = TTuvz1+Tz1.real
                # if (uz1.EQ.vz1) :
                # ruvz1=ziz1 * (2 * VAz1 * CF1z1 * (GRvz1(vz1, vz1))+ziz1)
                # else
                # ruvz1=ziz1 * dnp.sqrt(dabs(CF2z1 / CF1z1)) * (2 * VAz1 * CF1z1 * (GRvz1(uz1, vz1)))
                # end if
                # Rz1=ruvz1 * dconjg(ruvz1)
                # RRuvz1=RRuvz1+Rz1

                # TTz1=TTuvz1+RRuvz1

        # GFMEz1[ixz1, iyz1]=TTuvz1
        # @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
        # @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
        # write(210, * )fluxo1, TTuvz1

        file210.write(str(Ez1)+";"+str(TTuvz1)+"\n")
        En.append(Ez1)
        Tn.append(TTuvz1)
        file210.flush()
        os.fsync(file210)
        # file210.close()
        # write(320, * )Ez1, TTz1

    # write(210, * )Ez1, TTuvz1


# DO Jz1=0, 150
#  WRITE(360, 41)GFMEz1(0:150, Jz1)
# 41 FORMAT(151(F22 .16, 1 X))
# END DO

file210.close()

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

plt.figure(1)
plt.subplot(211)
#plt.scatter(En, Tn, linewidths=0, marker='.')
plt.plot(En, Tn, linewidth=0.4)
#plt.text(0, 4, r'$10000$ pontos, $\Delta E = 0.000055$', fontdict=font)
plt.xlabel('Energia', fontdict=font)
plt.ylabel('Transmissão', fontdict=font)

#plt.plotfile('EEzvsTTz1.dat',delimiter=';',cols=(0,1),names=('E','T'),marker='o')



plt.subplot(212)
plt.scatter(Dpx, Dpy)

plt.show()

#plt.plotfile('DISP01z1.dat',delimiter=';',cols=(0,1),names=('E','T'),linestyle="",marker="o")

