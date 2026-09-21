# distutils: language = c++
# cython: language_level=3
import cython

from libc.math cimport pi, cos, sin, exp

cdef complex cexp(double complex z):

    return exp(z.real)*(cos(z.imag)+1j*sin(z.imag))

cdef complex conj(double complex z):

    return z.conjugate() 

# expandvar
cdef void expandvar(complex[:] X,complex[:] mu,complex[:] Q12,complex[:] Q23,complex[:] Q31,
              complex[:] QC12,complex[:] QC23,complex[:] QC31):

    cdef long i, t

    t = 0
    
    for i in range(3): 
        mu[i] = X[t]
        t += 1
    for i in range(3):
        Q12[i] = X[t]
        t += 1
    for i in range(3):
        Q23[i] = X[t]
        t += 1
    for i in range(3):
        Q31[i] = X[t]
        t += 1
    
    for i in range(3):
        QC12[i] = X[t]
        t += 1
    for i in range(3):
        QC23[i] = X[t]
        t += 1
    for i in range(3):
        QC31[i] = X[t]
        t += 1


# vertex function
cdef void Vert(complex[:, :] M, long i0, long NK1, long NK2, long L):
    
    cdef complex mu[3]
    cdef complex Q12[3]
    cdef complex Q23[3]
    cdef complex Q31[3]
    cdef complex QC12[3]
    cdef complex QC23[3] 
    cdef complex QC31[3] 
    cdef long n1, n2, i, j, k, l
    cdef double k1, k2, k3
    cdef complex A[3][3]
    cdef complex B[3][3]
    cdef complex BC[3][3]
    cdef complex XM[21]

    XM[i0] = 1 + 0j

    expandvar(XM,mu,Q12,Q23,Q31,QC12,QC23,QC31)


    n1 = NK1//L
    n2 = NK1 % L

    k1 = 2*n1*pi/L
    k2 = 2*n2*pi/L
    k3 = -k1-k2
    
    cdef complex PHASE1ab[3]
    PHASE1ab = [1.0 + 0j, cexp(-1j*k3), cexp(1j*k2)]
    cdef complex PHASE1bc[3] 
    PHASE1bc = [cexp(1j*k3), 1.0 + 0j, cexp(-1j*k1)]
    cdef complex PHASE1ca[3]
    PHASE1ca = [cexp(-1j*k2), cexp(1j*k1), 1.0+0j]
    
    n1 = NK2//L
    n2 = NK2 % L

    k1 = 2*n1*pi/L
    k2 = 2*n2*pi/L
    k3 = -k1-k2
    
    cdef complex PHASE2ab[3]
    PHASE2ab = [1.0, cexp(-1j*k3), cexp(1j*k2)]
    cdef complex PHASE2bc[3]
    PHASE2bc = [cexp(1j*k3), 1.0, cexp(-1j*k1)]
    cdef complex PHASE2ca[3] 
    PHASE2ca = [cexp(-1j*k2), cexp(1j*k1), 1.0]
   
    for i in range(3):
            
        B[0][1] = B[0][1] - Q12[i] * PHASE2ab[i] / 2
        B[0][2] = B[0][2] + Q31[i] * conj(PHASE1ca[i])/2
        B[1][0] = B[1][0] + Q12[i] * conj(PHASE1ab[i])/2
        B[1][2] = B[1][2] -Q23[i] * PHASE2bc[i] / 2
        B[2][0] = B[2][0]- Q31[i] * PHASE2ca[i] / 2
        B[2][1] = B[2][1] + Q23[i] * conj(PHASE1bc[i]) / 2
        
        BC[0][1] = B[0][1] -QC12[i] * conj(PHASE1ab[i])/2
        BC[0][2] = B[0][2] + QC31[i] * PHASE2ca[i] / 2
        BC[1][0] = B[1][0] + QC12[i] * PHASE2ab[i] / 2
        BC[1][2] = B[1][2] - QC23[i] * conj(PHASE1bc[i])/2
        BC[2][0] = B[2][0] - QC31[i] * conj(PHASE1ca[i])/2
        BC[2][1] = B[2][1] + QC23[i] * PHASE2bc[i] / 2

    for i in range(3):
        A[i][i] = mu[i]

    for i in range(3):
        for j in range(3):
            M[i][j] = A[i][j]
            k = j + 3
            M[i][k] = B[i][j]
            k = i + 3
            M[k][j] = BC[j][i]
            k = i + 3
            l = j + 3
            M[k][l] = A[i][j]


cdef double Theta(E):

    cdef double res

    if E > 0:

        res = 1

    elif E == 0:

        res = -1

    else:

        res = 0

    return res

#def loopEval(complex[:,:] D0, complex[:] X, double h, double S, complex[:] Z, 
def loopEval(complex[:] X, double h, double S, 
             #complex[:] Z, 
             long L, 
             long flucDim, 
             long[:] irange, 
             complex[:, :, :] MK,
             double[:, :] EK,
             double[:] Sigma,
             long NQ,
             long NSYS,
             ):

    cdef long NK, NKQ, a, b, i, j, k, l, m, n, hfDim, kx, ky

    hfDim = flucDim//2

    cdef complex gamma1
    cdef complex gamma2
    cdef complex GAMMA_KQ[6][6]
    cdef complex GAMMA_QK[6][6]
    cdef complex SE

    SE = 0
    
    for NK in range(NSYS):
        
        NKQ = (NK + NQ)

        if NKQ < NSYS:

            for a in range(hfDim):

                i = irange[a]
                Vert(GAMMA_QK,i,NQ,NK,L)   

                b = a + hfDim
                j = irange[b]
                Vert(GAMMA_KQ,j,NK,NKQ,L)   
                
                for k in range(3):
                    for l in range(3,6):
                        gamma1 = 0j
                        gamma2 = 0j
                        for m in range(6):
                            for n in range(6):
                                gamma1 = gamma1 + MK[NKQ][m][k].conjugate() * GAMMA_QK[m][n] * MK[NK][n][l]
                                gamma2 = gamma2 + MK[NK][m][l].conjugate() * GAMMA_KQ[m][n] * MK[NKQ][n][k]
                        
                        #gamma2 *= Sigma[k]*Sigma[l]*(Theta(Sigma[k]*EK[NKQ][k])-Theta(Sigma[l]*EK[NK][l]))*(
                        #        Theta(Sigma[k]*EK[NKQ][k]-Sigma[l]*EK[NK][l])
                        #        )
                        SE = SE - gamma1 * gamma2
            
            for a in range(hfDim,flucDim):
                    
                i = irange[a]
                Vert(GAMMA_QK,i,NKQ,NK,L)   

                b = a - hfDim
                j = irange[b]
                Vert(GAMMA_KQ,j,NK,NKQ,L)   
                
                for k in range(3):
                    for l in range(3,6):
                        gamma1 = 0j
                        gamma2 = 0j
                        for m in range(6):
                            for n in range(6):
                                gamma1 = gamma1 + MK[NKQ][m][k].conjugate() * GAMMA_QK[m][n] * MK[NK][n][l]
                                gamma2 = gamma2 + MK[NK][m][l].conjugate() * GAMMA_KQ[m][n] * MK[NKQ][n][k]
                        
                        #gamma2 *= Sigma[k]*Sigma[l]*(Theta(Sigma[k]*EK[NKQ][k])-Theta(Sigma[l]*EK[NK][l]))*(
                        #        Theta(Sigma[k]*EK[NKQ][k]-Sigma[l]*EK[NK][l])
                        #        )
                        SE = SE - gamma1 * gamma2
    
    return (SE/L**2)
