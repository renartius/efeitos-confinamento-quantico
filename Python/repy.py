import numpy as np


def do(start, end, step=1):
    if start < end:
        while start <= (end-1) and step > 0:
            yield start
            start += step
    else:
        if start > end:
            while start >= (end+1) and step < 0:
                yield start
                start += step
        else:
            yield start


def invxz1(mm, nn):
    return np.linalg.inv(mm)


def mult_matrix_NxN(A, B):
    n = len(A)
    mult_mat = np.zeros((n,n),dtype='complex',order='F')
    for i in range(0,n):
        for j in range(0, n):
            for k in range(0, n):
                mult_mat[i,j] += A[i,k]*B[k,j]

    return np.transpose(mult_mat)



def transpose(a):
    n = len(a)
    t = np.zeros((n, n), dtype='complex', order='F')
    for i in range(0,n):
        for j in range(0,n):
            t[i,j] = a[j,i]

    return t


def mult_matrix_NxN2(A, B):
    n = len(A)
    #hacker_matrix(A,B)
    mult_mat = np.zeros((n,n),dtype='complex',order='F')
    for i in range(0,n):
        for j in range(0, n):
            for k in range(0, n):
                #print(A[i,k])
                #print(B[k,j])
                #print(A[i, k] * B[k, j])
                mult_mat[i,j] += A[i,k]*B[j,k]#*(i==j) + A[i,k]*B[k,j]*((i+j)==(n-1))
                #print(mult_mat[i,j])
    return mult_mat

def hacker_matrix(a,b):
    for linha in range(0,5):
        for coluna in range(0,5):
            print(str(linha)+","+str(coluna))
            c=0
            for j in range(0,5):
                c+=a[j,linha]*b[j,coluna]
            print("res: "+str(c))
    return 0