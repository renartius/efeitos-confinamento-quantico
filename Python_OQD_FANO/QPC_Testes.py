#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import repy as re
import os

NDz1 = 2
NWz1 = 3
MCz1 = 10
MQPC1z1 = 10
MQPC2z1 = 10
MOQDz1 = 20
NMz1 = 2*MCz1+MQPC1z1+MOQDz1+MQPC2z1

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

DISPOz1 = np.zeros((NMz1, NDz1), dtype='float',order='F')
GFMEz1 = np.zeros((NMz1, NDz1), dtype='complex',order='F')
HCz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
HQPC1z1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
HQPC2z1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
HOQDz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
HGz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
QW1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')

MDUXz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
MDUXB1z1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
MDUXB2z1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
CLz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
UUz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
GUz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
CLXz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
CRXz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
VHXz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
VHXB1z1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
VHXB2z1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
GTz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
GRz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
GIz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
XXz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
XX1z1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
XXAz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
XX1Az1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
HIz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
XIz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
GTvz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
GRvz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')

HICz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
HIXz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
XIXz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
GIICz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')
GIXz1 = np.zeros((NDz1, NDz1), dtype='complex',order='F')

file210 = open("EEzvsTTz1.dat","w")
file340 = open("DISP01z1.dat","w")
file360 = open("GvsFMEz1.dat","w")

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
EAz1 = 2 * (VAz1) * (2)
EBz1 = 2 * (VBz1) * (2) + 0.1
VHz1 = abs(-np.sqrt((VAz1 * VBz1)))


# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# HC: hamiltoniano tipo tight binding da cadeia transversal do contato.
# HC(i, j): elementos de matriz do hamiltoniano.i: columna, j: fila.
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
for ixz1 in range(0,NDz1):
    for jxz1 in range(0,NDz1):
        HCz1[jxz1, ixz1] = z0z1
#@ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
for imz1 in range(0,NDz1):
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
HCz1 = HGz1

WBz1 = NDz1
yyz1 = 0
for iyz1 in range(1, WBz1+1):
    yy1z1 = yyz1 + iyz1
    for imz1 in range(1, 2):
        HCz1[yy1z1 + (imz1 - 1) * NDz1-1, yy1z1 + (imz1 - 1) * NDz1-1] = EAz1 * zrz1

# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** QPC1 ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
HQPC1z1 = HGz1

WBz1 = NWz1

# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** QPC2 ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
HQPC2z1 = HQPC1z1

# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** OQD ** ** ** ** ** ** ** **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
HOQDz1 = HGz1

WBz1 = NWz1        # se for usar QPC, tirar comentario e comentar linha de baixo
# WBz1 = NDz1
yyz1 = 4            # se for usar QPC, yyz1 = 4


# @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @

# @ @ @ @ @ @ @ @ @ @ @ @ @IMPUREZA CONTROLAVEL @ @ @ @ @ @ @ @ @ @ @ @ @ @
HICz1 = HGz1

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

    for ixz1 in range(0, 1):
        vv1z1 = 1
        Ev1z1 = 2.0 * VAz1 - 2.0 * VAz1 * np.cos(piz1 * vv1z1 / (NDz1 + 1))
        piz1 = 3.141592654

        # Ez1 = 0.0 + 1e-20 + ixz1 * ((4) * VAz1 + Ev1z1 - 0.00001) / 90000
        # #Ez1 = 0.01339
        Ez1 = 0.0 + 1e-20 + ixz1 * 0.00055

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
                #UUz1[jnz1-1, inz1-1] = CC1z1 * np.sin(piz1 * vvz1 * inz1 / (NDz1 + 1))
                UUz1[jnz1 - 1, inz1 - 1] = CC1z1 * np.sin(piz1 * vvz1 * inz1 / (NDz1 + 1))

        for ivz1 in range(1, NDz1+1):
            Evz1 = 2.0 * VAz1 - 2.0 * VAz1 * np.cos(piz1 * ivz1 / (NDz1 + 1))
            if (Ez1 >= 0) and (Ez1 < Evz1):
                CAz1 = np.cos(np.arccos((Ez1 - Evz1) / (2. * VAz1) + 1.))*zrz1 - np.sin(np.arccos((Ez1 - Evz1) / (2. * VAz1) + 1.))*ziz1
            else:
                CAz1 = np.cos(np.arccos((Ez1 - Evz1) / (2. * VAz1) - 1.))*zrz1 - np.sin(np.arccos((Ez1 - Evz1) / (2. * VAz1) - 1.))*ziz1
            GUz1[ivz1-1, ivz1-1] = (1. / VAz1) * CAz1

        print(np.matmul(np.transpose(UUz1),GUz1))
        print("")
        x = np.zeros((NDz1, NDz1), dtype='complex', order='F')
        x=np.matmul(GUz1, np.transpose(UUz1))
        print(x[0,1])
        print("")

        #CLz1 = re.mult_matrix_NxN2(UUz1, re.mult_matrix_NxN(GUz1, np.transpose(UUz1)))

        #CLz1 = np.matmul(UUz1, np.matmul(GUz1, np.transpose(UUz1)))
        CLz1 = np.matmul(UUz1, x)
        print(CLz1)
        print("")
        print(CLz1[0, 1])

        y = np.zeros((NDz1, NDz1), dtype='complex', order='F')
        print("calc ")
        print(UUz1[0, 0] * x[0, 1]+UUz1[0, 1] * x[1, 1])
