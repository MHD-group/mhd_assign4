#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Created On  : 2023-04-03 00:24
# Last Modified : 2023-05-20 02:18
# Copyright © 2023 myron <yh131996@mail.ustc.edu.cn>
#
# Distributed under terms of the MIT license.


import numpy as np
from numpy import arange, cos, sqrt, sin, abs
from numpy import pi as π
from matplotlib import pyplot as plt
import argparse

# β = 2 γ = 1.6667 accroding to xlsx
_β = 2
_γ = 1.6667
_μ = 1
_Hx = 5

def W2U(W):
    U = W.copy()
    U[:, 1] = W[:, 0]*(W[:, 2]**2 + W[:, 3]**2 + W[:, 4]**2)\
        + W[:, 5]**2 + W[:, 6]**2 + (_β*W[:, 1])/(_γ - 1)
    U[:, 2] = W[:, 0]*W[:, 2]
    U[:, 3] = W[:, 0]*W[:, 3]
    U[:, 4] = W[:, 0]*W[:, 4]
    return U

def U2W(U):
    W = U.copy()
    W[:, 2] = U[:, 2]/U[:, 0]
    W[:, 3] = U[:, 3]/U[:, 0]
    W[:, 4] = U[:, 4]/U[:, 0]
    W[:, 1] = (U[:, 1] - (W[:, 0]*(W[:, 2]**2 + W[:, 3]**2 + W[:, 4]**2))\
        - W[:, 5]**2 - W[:, 6]**2) * (_γ - 1) / _β
    return W

# wave's init value (non-conservative)
# case 0: 2.1 fast
# case 1: 2.1 slow
# case 2: 2.2 fast
# case 3: 2.3 slow
def init(x, type="W", case=0):
    W = np.array([\
        [  2.121,  4.981, -13.27, -0.163, -0.6521, 2.572, 10.29],\
        [      1,      1,  -15.3,      0,       0,     1,     4],\
        [  2.219, 0.4442, 0.5048, 0.0961,  0.0961,     1,     1],\
        [      1,    0.1,-0.9225,      0,       0,     1,     1],\
        [  3.896,  305.9,      0, -0.058,  -0.226, 3.951,  15.8],\
        [      1,      1,  -15.3,      0,       0,     1,     4],\
        [  3.108, 1.4336,      0, 0.2633,  0.2633,   0.1,   0.1],\
        [      1,    0.1,-0.9225,      0,       0,     1,     1],\
        ], float)
    N = x.size
    tmp = np.zeros((7, N), dtype=x.dtype)
    if case <=1:
        conds = [x <= -0, x > 0]
    else:
        conds = [x <= 0.2, x > 0.2]
    if type == "W":
        funcs = W[case*2:case*2+2,:].T
    elif type == "U":
        U = W2U(W)
        funcs = U[case*2:case*2+2,:].T
    for i in range(7):
        tmp[i] = np.piecewise(x, conds, funcs[i])
    return tmp.T

# wave's ref shape from excel
def funcRef(x, C=1, t=0):
    conds = [x < -2.633*t,\
             np.logical_and(x < -1.636*t, x >= -2.633*t),\
             np.logical_and(x < 1.529*t, x >= -1.636*t),\
             np.logical_and(x < 2.480*t, x >= 1.529*t),\
             x >= 2.480*t]
    func_ρ = [0.445,\
             lambda x : (x - (-2.633*t)) * (0.345 - 0.445)/(-1.636*t +2.633*t) + 0.445,\
             0.345,\
             1.304,\
             0.500]
    func_m = [0.311,\
             lambda x : (x - (-2.633*t)) * (0.527 - 0.311)/(-1.636*t +2.633*t) + 0.311,\
             0.527,\
             1.994,\
             0.000]
    func_E = [8.928,\
             lambda x : (x - (-2.633*t)) * (6.570 - 8.928)/(-1.636*t +2.633*t) + 8.928,\
             6.570,\
             7.691,\
             1.428]
    return [np.piecewise(x, conds, func_ρ), np.piecewise(x, conds, func_E), np.piecewise(x, conds, func_m)]


def U2F(U):
    F = U.copy()
    W = U2W(U)
    F[:, 0] = U[:, 2]
    F[:, 1] = U[:, 2] * (W[:,2]**2 + W[:,3]**2 + W[:,4]**2\
              + (_γ*_β*W[:,1])/(U[:,0] * (_γ - 1)))\
              + 2*(U[:,5]**2*W[:,2] + U[:,6]**2*W[:,2] -\
              _Hx * U[:,5] * W[:,3] - _Hx * U[:,6] * W[:, 4])
    F[:, 2] = U[:, 2] * W[:, 2] + 0.5*(_β * W[:,1] +\
              U[:, 5]**2 + U[:, 6]**2)
    F[:, 3] = U[:, 2] * W[:, 3] - _Hx * U[:, 5]
    F[:, 4] = U[:, 2] * W[:, 4] - _Hx * U[:, 6]
    F[:, 5] = U[:, 5] * W[:, 2] - _Hx * W[:, 3]
    F[:, 6] = U[:, 6] * W[:, 2] - _Hx * W[:, 4]
    return F

def U2A(U):
    ga = _γ
    Hx = _Hx
    A = np.zeros((np.size(U, 0), 7, 7), float)
    #A[:,0,0] = 0
    #A[:,0,1] = 0
    A[:,0,2] = 1
    #A[:,0,3] = 0
    #A[:,0,4] = 0
    #A[:,0,5] = 0
    #A[:,0,6] = 0
    A[:,1,0] = (2*Hx*U[:,0]*(U[:,5]*U[:,3]+U[:,6]*U[:,4])+U[:,2]*(U[:,0]*(-ga*U[:,1]+(ga-2)*U[:,5]**2+(ga-2)*U[:,6]**2)+2*(ga-1)*U[:,3]**2+2*(ga-1)*U[:,4]**2)+2*(ga-1)*U[:,2]**3)/U[:,0]**3
    A[:,1,1] = (ga*U[:,2])/U[:,0]
    A[:,1,2] = -((-ga*U[:,1]*U[:,0]+(ga-2)*U[:,5]**2*U[:,0]+(ga-2)*U[:,6]**2*U[:,0]+3*(ga-1)*U[:,2]**2+(ga-1)*U[:,3]**2+(ga-1)*U[:,4]**2)/U[:,0]**2)
    A[:,1,3] = -((2*(Hx*U[:,5]*U[:,0]+(ga-1)*U[:,2]*U[:,3]))/U[:,0]**2)
    A[:,1,4] = -((2*(Hx*U[:,6]*U[:,0]+(ga-1)*U[:,2]*U[:,4]))/U[:,0]**2)
    A[:,1,5] = -((2*(Hx*U[:,3]+(ga-2)*U[:,5]*U[:,2]))/U[:,0])
    A[:,1,6] = -((2*(Hx*U[:,4]+(ga-2)*U[:,6]*U[:,2]))/U[:,0])
    A[:,2,0] = ((ga-3)*U[:,2]**2+(ga-1)*(U[:,3]**2+U[:,4]**2))/(2*U[:,0]**2)
    A[:,2,1] = (ga-1)/2
    A[:,2,2] = -(((ga-3)*U[:,2])/U[:,0])
    A[:,2,3] = (U[:,3]-ga*U[:,3])/U[:,0]
    A[:,2,4] = (U[:,4]-ga*U[:,4])/U[:,0]
    A[:,2,5] = (ga-2)*(-U[:,5])
    A[:,2,6] = (ga-2)*(-U[:,6])
    A[:,3,0] = -((U[:,2]*U[:,3])/U[:,0]**2)
    #A[:,3,1] = 0
    A[:,3,2] = U[:,3]/U[:,0]
    A[:,3,3] = U[:,2]/U[:,0]
    #A[:,3,4] = 0
    A[:,3,5] = -Hx
    #A[:,3,6] = 0
    A[:,4,0] = -((U[:,2]*U[:,4])/U[:,0]**2)
    #A[:,4,1] = 0
    A[:,4,2] = U[:,4]/U[:,0]
    #A[:,4,3] = 0
    A[:,4,4] = U[:,2]/U[:,0]
    #A[:,4,5] = 0
    A[:,4,6] = -Hx
    A[:,5,0] = (Hx*U[:,3]-U[:,5]*U[:,2])/U[:,0]**2
    #A[:,5,1] = 0
    A[:,5,2] = U[:,5]/U[:,0]
    A[:,5,3] = -(Hx/U[:,0])
    #A[:,5,4] = 0
    A[:,5,5] = U[:,2]/U[:,0]
    #A[:,5,6] = 0
    A[:,6,0] = (Hx*U[:,4]-U[:,6]*U[:,2])/U[:,0]**2
    #A[:,6,1] = 0
    A[:,6,2] = U[:,6]/U[:,0]
    #A[:,6,3] = 0
    A[:,6,4] = -(Hx/U[:,0])
    #A[:,6,5] = 0
    A[:,6,6] = U[:,2]/U[:,0]
    return A


def shift1(X):
    return np.pad(np.roll(X, 1, axis=0)[1:-1,:,:], ((1,), (0,), (0,)), 'edge')
def shift_sub1(X):
    return X - shift1(X)
def shift_ave1(X):
    return 0.5*(X + shift1(X))

# still need to be done
def Lax_u_n(u, C=0.5, t=100):
    #print('calling upwind, ', w, γ, C, t)
    N = np.size(u,0)
    print(N)
    f = U2F(u)
    a = U2A(u)
    tmp_u = np.expand_dims(np.pad(u, ((1,), (0,)), 'edge'), [0, -1]).repeat(2, axis=0)
    F = np.expand_dims(np.pad(f, ((1,), (0,)), 'edge'), [-1])
    A = np.pad(a, ((1,), (0,), (0,)), 'edge')
    print("----------")
    print(tmp_u.shape)
    for n in range(2):
        cur = n%2
        nex = (n%2 + 1)%2
        F= np.expand_dims(U2F(tmp_u[cur,:,:,0]), [-1])
        A = U2A(tmp_u[cur,:,:,0])
        if n == 0:
            print(t)
            print(F.shape)
            print(A.shape)
        dF = shift_sub1(F)
        ddF = shift_sub1(dF)
        aU = shift_ave1(tmp_u[cur,:,:,:])
        A = U2A(aU[:,:,0])
        Lax_term = A@dF
        Lax_sub = shift_sub1(Lax_term)
        tmp_u[nex,:,:,:] = tmp_u[cur,:,:,:] - 0.5*C*shift1(ddF) + 0.5*C*C*shift1(Lax_sub)

#        for i in range(N):
#            I = i+1
#            #Am = U2A(0.5*(tmp_u[cur,I-1:I,:,0] + tmp_u[cur,I:I+1,:,0]))
#            #Ap = U2A(0.5*(tmp_u[cur,I:I+1,:,0] + tmp_u[cur,I+1:I+2,:,0]))
#            tmp_u[nex, I:I+1, :, :] = tmp_u[cur, I:I+1, :, :]\
#                -0.5*C*ddF[I+1:I+2, :, :]\
#                +0.5*C*C*(A[I+1:I+2,:,:]@dF[I+1:I+2,:,:]\
#                -A[I:I+1]@dF[I:I+1,:,:])
        result = tmp_u[nex,1:-1,:]
        tmp_u = np.pad(tmp_u[:,1:-1,:,:], ((0,),(1,),(0,),(0,)), 'edge')
    print("result:", result.shape)
    return result

def Upwind_u(u, C=0.5, t=100):
    #print('calling upwind, ', w, γ, C, t)
    γ = _γ
    N = u[:,0].size
    tmp_u = np.expand_dims(np.pad(u, ((1,), (0,)), 'edge'), [0, -1]).repeat(2, axis=0)
    for n in range(t):
        cur = n%2
        nex = (n%2 + 1)%2
        c_u = tmp_u[cur,:,:,:]
        #dia_ = U2λ(c_u, γ)
        #R = U2R(c_u, γ)
        #L = U2L(c_u, γ)
        #A = R@dia_@L
        A = U2A(c_u[:,:,0])
        vals, vecs = np.linalg.eig(A)
        dia_ = vecs.copy()
        for ii in range(N+2):
            dia_[ii, :, :] = np.diag(vals[ii, :])
        R = vecs
        L = np.linalg.inv(vecs)
        Rl = np.pad(np.roll(R, 1, axis=0)[1:-1,:,:], ((1,),(0,),(0,)), 'edge')
        Rr = np.pad(np.roll(R, -1, axis=0)[1:-1,:,:], ((1,),(0,),(0,)), 'edge')
        Ll = np.pad(np.roll(L, 1, axis=0)[1:-1,:,:], ((1,),(0,),(0,)), 'edge')
        Lr = np.pad(np.roll(L, -1, axis=0)[1:-1,:,:], ((1,),(0,),(0,)), 'edge')
        dia_l = np.pad(np.roll(dia_, 1, axis=0)[1:-1,:,:], ((1,),(0,),(0,)), 'edge')
        dia_r = np.pad(np.roll(dia_, -1, axis=0)[1:-1,:,:], ((1,),(0,),(0,)), 'edge')
        pos_dl = np.where(dia_l>=0, dia_l, 0)
        neg_dl = np.where(dia_l<0, dia_l, 0)
        pos_dr = np.where(dia_r>=0, dia_r, 0)
        neg_dr = np.where(dia_r<0, dia_r, 0)
        pos_d = np.where(dia_>=0, dia_, 0)
        neg_d = np.where(dia_<0, dia_, 0)
        up = c_u - np.pad(np.roll(c_u, 1, axis=0)[1:-1,:,:], ((1,),(0,),(0,)), 'edge')
        um = np.pad(np.roll(c_u, -1, axis=0)[1:-1,:,:], ((1,),(0,),(0,)), 'edge') - c_u
        #tmp_u[nex, :, :, :] =  c_u \
        #        - C * ((0.5*Rl@pos_dl@Ll+0.5*R@pos_dl@L)@up\
        #        + (0.5*Rr@neg_dr@Lr + 0.5*R@neg_dr@L)@um)
        tmp_u[nex, :, :, :] =  c_u \
                - C * ((0.5*Rl@pos_dl@Ll+0.5*R@pos_d@L)@up\
                + (0.5*Rr@neg_dr@Lr + 0.5*R@neg_d@L)@um)
        result = tmp_u[nex,1:-1,:,:]
    return result

def shift(u, i=1):
    if i == 0:
        return u
    else:
        return np.pad(np.roll(u, i, axis=0)[1:-1,:,:], ((1,),(0,),(0,)), 'edge')

def get_RL(c_u):
    A = U2A(c_u[:,:,0])
    N = c_u[:,0,0].size
    vals, vecs = np.linalg.eig(A)
    dia_ = vecs.copy()
    for ii in range(N):
        dia_[ii, :, :] = np.diag(vals[ii, :])
    R = vecs
    L = np.linalg.inv(vecs)
    return R,dia_,L

def div_pn(A):
    return np.where(A>=0, A, 0), np.where(A<0, A, 0)

from numpy import logical_and as band
from numpy import logical_not as bnot
from numpy import logical_or as bor

def minmod(A, B):
    c1 = np.where(A > B, True, False)
    c2 = np.where(A > 0, True, False)
    c3 = np.where(B > 0, True, False)
    result = np.where(band(bnot(c1), c2), A, 0)
    result = np.where(band(c1, c2), B, result)
    result = np.where(band(c1, bnot(c2)), A, result)
    result = np.where(band(bnot(c1), bnot(c3)), B, result)
    return result

def TVD_u(u, C=0.5, t=100):
    #print('calling upwind, ', w, γ, C, t)
    γ = _γ
    N = u[:,0].size
    tmp_u = np.expand_dims(np.pad(u, ((1,), (0,)), 'edge'), [0, -1]).repeat(2, axis=0)
    for n in range(t):
        cur = n%2
        nex = (n%2 + 1)%2
        c_u = tmp_u[cur,:,:,:]
        R,dia_,L = get_RL(c_u)
        Rl = shift(R, 1)
        Rr = shift(R, -1)
        Ll = shift(L, 1)
        Lr = shift(L, -1)
        dia_l = shift(dia_, 1)
        dia_r = shift(dia_, -1)
        pos_dl, neg_dl = div_pn(dia_l)
        pos_dr, neg_dr = div_pn(dia_r)
        pos_d, neg_d = div_pn(dia_)
        up = c_u - shift(c_u, 1)
        um = shift(c_u, -1) - c_u
        pos_A = 0.5*Rl@pos_dl@Ll + 0.5*R@pos_d@L
        neg_A = 0.5*Rr@neg_dr@Lr + 0.5*R@neg_d@L
        #pos_A = Rl@pos_dl@Ll
        #neg_A = Rr@neg_dr@Lr
        #pos_A = 1*R@pos_d@L
        #neg_A = 1*R@neg_d@L
        tmp_u[nex, :, :, :] =  c_u \
                - C * (pos_A@up + neg_A@um)\
                - 0.5 * C * (1 - C) *\
                ( minmod(pos_A@up, shift(pos_A@up, -1)) -\
                minmod(pos_A@up, shift(pos_A@up, 1)))\
                - 0.5 * C * (1 - C) *\
                ( minmod(shift(neg_A@um, 1), shift(neg_A@um, 0)) -\
                minmod(shift(neg_A@um, -1), shift(neg_A@um, 0)))
        result = tmp_u[nex,1:-1,:,:]
    return result

def Lax_u(u, C=0.5, t=100):
    #print('calling upwind, ', w, γ, C, t)
    N = np.size(u,0)
    f = U2F(u)
    a = U2A(u)
    tmp_u = np.expand_dims(np.pad(u, ((1,), (0,)), 'edge'), [0, -1]).repeat(2, axis=0)
    F = np.expand_dims(np.pad(f, ((1,), (0,)), 'edge'), [-1])
    A = np.pad(a, ((1,), (0,), (0,)), 'edge')
    for n in range(t):
        cur = n%2
        nex = (n%2 + 1)%2
        F = np.expand_dims(U2F(tmp_u[cur,:,:,0]), [-1])
        A = U2A(tmp_u[cur,:,:,0])
        if n == 0:
            print(t)
            print(F.shape)
            print(A.shape)
        for i in range(N):
            I = i+1
            tmp_u[nex, I:I+1, :, :] = tmp_u[cur, I:I+1, :, :]\
                -0.5*C*(F[I+1:I+2, :, :] - F[I-1:I, :, :])\
                +0.5*C*C*(0.5*(A[I+1:I+2,:,:] + A[I:I+1,:,:])@(F[I+1:I+2,:,:] - F[I:I+1,:,:])\
                -0.5*(A[I:I+1,:,:] + A[I-1:I,:,:])@(F[I:I+1,:,:] - F[I-1:I,:,:]))
        result = tmp_u[nex,1:-1,:]
        tmp_u = np.pad(tmp_u[:,1:-1,:,:], ((0,),(1,),(0,),(0,)), 'edge')
    print("result:", result.shape)
    return result


if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description="calculate X to the power of Y")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument("-x", "--resolution", default=0.01, type=float, help="length of Δx")
    parser.add_argument("-n", "--numbers", default=0, type=int, help="length of Δx")
    parser.add_argument("-C", "--ratio", default=0.5, type=float, help="Δt/Δx")
    parser.add_argument("-s", "--start", default=-1, type=float, help="f(x) when t=0")
    parser.add_argument("-e", "--end", default=1, type=float, help="f(x) when t=0")
    parser.add_argument("-m", "--methods", default="Upwind,LaxWendroff", type=str, help="methods")
    parser.add_argument("-t", "--times", default="0,0.1", type=str, help="time")
    parser.add_argument("-case", "--case", default=0, type=int, help="case")
    parser.add_argument("-o", "--output", default="W", type=str, help="output format")
    parser.add_argument("-i", "--input", default="U", type=str, help="intput format")
    parser.add_argument("-w", "--watch", default="0.1,0.5", type=str, help="intput format")
    parser.add_argument("-y", "--ylim", default="1.5,1.5,1.6,1.7,1.8,1.9,1.9", type=str, help="intput format")
    args = parser.parse_args()

    print(args.watch)
    Ts = [float(idx) for idx in args.times.split(',')]
    wt = [float(idx) for idx in args.watch.split(',')]
    print(wt)
    ylims = [float(idx) for idx in args.ylim.split(',')]
    methods = args.methods.split(',')
    if args.numbers != 0:
        res = (args.end-args.start)/args.numbers
    else:
        res = args.resolution
    # Δx: args.resolution
    x = np.arange(args.start, args.end, res)
    # C is Δt/Δx
    C = args.ratio
    # Δt
    t = C * res

    case = args.case
    print("running case ", case)
    u = init(x, args.input, case)
    w = U2W(u)
    print(w[1])
    print(u[1])
    F = U2F(u)
    print(F[1])
    # A = U2A(u)
    # print("A:", A.shape)

    ## show init stats
    #print(u.shape)
    #fig, axs = plt.subplots(7,
    #                        1,
    #                        figsize=(40, 12))
    #print(range(np.size(u, 1)))
    #for i in range(np.size(u, 1)):
    #    print(i)
    #    axs[i].plot(x, u[:,i])
    #plt.show()

    fig, axs = plt.subplots(7,
                            1,#len(methods),
                            figsize=(40, 30))
    for (T, i) in zip(Ts, range(len(Ts))):
        for (method, j) in zip(methods, range(len(methods))):
            n_t = int(T/t)
            if n_t == 0:
                output = u
            else:
                if method == "Upwind":
                    output = Upwind_u(u, C, n_t)
                elif method == "LaxWendroff":
                    output = Lax_u_n(u, C, n_t)
                elif method == "TVD":
                    output = TVD_u(u, C, n_t)
                else:
                    print("error input function")
            print(j)
            if args.output == "W":
                output = U2W(output)
            axs[0].plot(x, output[:, 0], linewidth=1, marker="o", markeredgewidth=0.6, markersize=4, markerfacecolor="none")
            axs[1].plot(x, output[:, 1], linewidth=1, marker="o", markeredgewidth=0.6, markersize=4, markerfacecolor="none")
            axs[2].plot(x, output[:, 2], linewidth=1, marker="o", markeredgewidth=0.6, markersize=4, markerfacecolor="none")
            axs[3].plot(x, output[:, 3], linewidth=1, marker="o", markeredgewidth=0.6, markersize=4, markerfacecolor="none")
            axs[4].plot(x, output[:, 4], linewidth=1, marker="o", markeredgewidth=0.6, markersize=4, markerfacecolor="none")
            axs[5].plot(x, output[:, 5], linewidth=1, marker="o", markeredgewidth=0.6, markersize=4, markerfacecolor="none")
            axs[6].plot(x, output[:, 6], linewidth=1, marker="o", markeredgewidth=0.6, markersize=4, markerfacecolor="none")

    axs[0].set_xlim([wt[0], wt[1]])
    axs[1].set_xlim([wt[0], wt[1]])
    axs[2].set_xlim([wt[0], wt[1]])
    axs[3].set_xlim([wt[0], wt[1]])
    axs[4].set_xlim([wt[0], wt[1]])
    axs[5].set_xlim([wt[0], wt[1]])
    axs[6].set_xlim([wt[0], wt[1]])

    axs[0].set_ylabel("ρ")
    axs[1].set_ylabel("E")
    axs[2].set_ylabel("ρv_x")
    axs[3].set_ylabel("ρv_y")
    axs[4].set_ylabel("ρv_x")
    axs[5].set_ylabel("H_y")
    axs[6].set_ylabel("H_z")

    axs[0].set_ylim([ylims[0], ylims[1]])
    axs[1].set_ylim([ylims[2], ylims[3]])
    axs[2].set_ylim([ylims[4], ylims[5]])
    axs[3].set_ylim([ylims[6], ylims[7]])
    axs[4].set_ylim([ylims[8], ylims[9]])
    axs[5].set_ylim([ylims[10], ylims[11]])
    axs[6].set_ylim([ylims[12], ylims[13]])

    #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #fig.legend(lines, labels, scatterpoints = 1)
    #axs[0].legend(loc="upper right")
    #axs[1].legend(loc="upper right")
    #axs[2].legend(loc="upper right")
    #axs[3].legend(loc="upper right")
    #axs[4].legend(loc="upper right")
    #axs[5].legend(loc="upper right")
    #axs[6].legend(loc="upper right")
    print(wt[0], wt[1])
    print(ylims)
    #plt.savefig('../figures/case1_fast_upwind_TVD.pdf', bbox_inches='tight')
    #plt.savefig('../figures/case1_slow_upwind_TVD.pdf', bbox_inches='tight')
    plt.savefig('../figures/case3_fast_upwind_TVD.pdf', bbox_inches='tight')
    #plt.show()

