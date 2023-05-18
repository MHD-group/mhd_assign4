#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Created On  : 2023-04-03 00:24
# Last Modified : 2023-04-28 22:08
# Copyright © 2023 myron <yh131996@mail.ustc.edu.cn>
#
# Distributed under terms of the MIT license.


import numpy as np
from numpy import arange, cos, sqrt, sin, abs
from numpy import pi as π
from matplotlib import pyplot as plt
import argparse

# wave's init shape
def init_w(x):
    N = x.size
    tmp = np.zeros((3, N), dtype=x.dtype)
    conds = [x <= -0, x > 0]
    funcs = [[0.445, 0.5], [0.311, 0], [8.928, 1.4275]]
    #funcs = [[0.945, 0.5], [0.311, 0], [8.928, 1.4275]]
    for i in range(3):
        tmp[i] = np.piecewise(x, conds, funcs[i])
    #print(tmp.shape)
    #print(tmp[:,0:1].shape)
    #print("----------")
    tmp1 = tmp.transpose(1, 0)[:,:,np.newaxis]
    #print(tmp1.shape)
    #print(tmp1[:,1,0].shape)
#    print(tmp1[0:1,:,np.newaxis].shape)
    return tmp.transpose(1,0)[:,:,np.newaxis]

# conservation
def w2U(w, γ=1.4):
    u=w.copy()
    u[:, 0, 0] = w[:, 0, 0]
    u[:, 1, 0] = w[:, 1, 0] / u[:, 0, 0]
    u[:, 2, 0] = (γ-1) * (w[:,2,0] - 0.5 * u[:,0,0] * u[:,1,0]**2)
    return u

def U2w(u, γ=1.4):
    w=u.copy()
    w[:, 0, 0] = u[:, 0, 0]
    w[:, 1, 0] = u[:, 1, 0] * u[:, 0, 0]
    w[:, 2, 0] = u[:, 2, 0] / (γ - 1) + 0.5*u[:, 0, 0]*u[:, 1, 0]**2
    return w

# vector to Matrix
def w2A(w, γ=1.4):
    U = w2U(w, γ)
    ρ = w[:,0,0]
    m = w[:,1,0]
    u = U[:,1,0]
    E = w[:,2,0]
    A = np.zeros((ρ.size, 3, 3), float)
    A[:,0,1] = 1
    A[:,1,0] = 0.5*(u**2)*(γ-3)
    A[:,1,1] = -u*(γ-3)
    A[:,1,2] = (γ-1)
    A[:,2,0] = (γ-1)*(u**3)-γ*u*E/ρ
    A[:,2,1] = (γ/ρ)*E - 1.5*(γ-1)*u**2
    A[:,2,2] = γ*u
    return A

def w2F(w, γ=1.4):
    U = w2U(w, γ)
    u = U[:,1:2,:]
    p = U[:,2:3,:]
    sub = np.concatenate((np.zeros(p.shape, float),\
            p, p*u), axis=1)
    F = u*w+sub
    #print(F.shape)
    return F

def w2R(w, γ=1.4):
    U = w2U(w, γ)
    ρ = w[:,0,0]
    m = w[:,1,0]
    E = w[:,2,0]
    u = U[:,1,0]
    p = U[:,2,0]
    a = sqrt(γ*p/ρ)
    H = (a**2)/(γ-1) + 0.5*u**2

    R = np.zeros((ρ.size, 3, 3), float)
    R[:,0,0] = 1
    R[:,0,1] = 1
    R[:,0,2] = 1
    R[:,1,0] = u-a
    R[:,1,1] = u
    R[:,1,2] = u+a
    R[:,2,0] = H-u*a
    R[:,2,1] = 0.5*u**2
    R[:,2,2] = H+u*a
    return R

# vector to Matrix
def w2L(w, γ=1.4):
    U = w2U(w, γ)
    ρ = w[:,0,0]
    m = w[:,1,0]
    E = w[:,2,0]
    u = U[:,1,0]
    p = U[:,2,0]
    a = sqrt(γ*p/ρ)
    H = γ*p/(ρ*(γ-1)) + 0.5*u**2
    K = 0.5*(γ-1)*ρ/(γ*p)

    L = np.zeros((ρ.size, 3, 3), float)
    L[:,0,0] = K * 0.5*u*(u+2*a/(γ-1))
    L[:,0,1] = K * -(u+a/(γ-1))
    L[:,0,2] = K * 1
    L[:,1,0] = K * 2*(H-u**2)
    L[:,1,1] = K * 2*u
    L[:,1,2] = K * -2
    L[:,2,0] = K * 0.5*u*(u-2*a/(γ-1))
    L[:,2,1] = K * -(u-a/(γ-1))
    L[:,2,2] = K * 1
    return L

# vector to Matrix
def U2λ(U, γ=1.4):
    ρ = U[:,0,0]
    u = U[:,1,0]
    p = U[:,2,0]
    a = sqrt(γ*p/ρ)
    λ = np.zeros((ρ.size, 3, 3), float)
    λ[:,0,0] = u-a
    λ[:,1,1] = u
    λ[:,2,2] = u+a
    return λ

def w2λ(w, γ=1.4):
    return U2λ(w2U(w, γ), γ)


# non conservation
# vector to Matrix
def U2R(U, γ=1.4):
    ρ = U[:,0,0]
    u = U[:,1,0]
    p = U[:,2,0]
    R = np.zeros((ρ.size, 3, 3), float)
    a = sqrt(γ*p/ρ)
    R[:,0,0] = 0.5/(a**2)
    R[:,0,1] = 1/(a**2)
    R[:,0,2] = 0.5/(a**2)
    R[:,1,0] = -0.5/(ρ*a)
    R[:,1,1] = 0
    R[:,1,2] = 0.5/(ρ*a)
    R[:,2,0] = 0.5
    R[:,2,1] = 0
    R[:,2,2] = 0.5
    return R

# vector to Matrix
def U2L(U, γ=1.4):
    ρ = U[:,0,0]
    u = U[:,1,0]
    p = U[:,2,0]
    L = np.zeros((ρ.size, 3, 3), float)
    a = sqrt(γ*p/ρ)
    L[:,0,0] = 0
    L[:,0,1] = -ρ*a
    L[:,0,2] = 1
    L[:,1,0] = a**2
    L[:,1,1] = 0
    L[:,1,2] = -1
    L[:,2,0] = 0
    L[:,2,1] = ρ*a
    L[:,2,2] = 1
    return L

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

def Upwind(w, γ=1.4, C=0.5, t=100):
    #print('calling upwind, ', w, γ, C, t)
    u = w2U(w, γ)
    N = u[:,0,0].size
    tmp_u = np.expand_dims(np.pad(u,((1,),(0,),(0,)), 'edge'), 0).repeat(2, axis=0)
    for n in range(t):
        cur = n%2
        nex = (n%2 + 1)%2
        c_u = tmp_u[cur,:,:,:]
        dia_λ = U2λ(c_u, γ)
        pos = np.where(dia_λ>0, dia_λ, 0)
        neg = np.where(dia_λ<=0, dia_λ, 0)
        R = U2R(c_u, γ)
        L = U2L(c_u, γ)
        up = c_u - np.pad(np.roll(c_u, 1, axis=0)[1:-1,:,:], ((1,),(0,),(0,)), 'edge')
        um = np.pad(np.roll(c_u, -1, axis=0)[1:-1,:,:], ((1,),(0,),(0,)), 'edge') - c_u
        tmp_u[nex, :, :, :] =  c_u \
                - C * (R@pos@L@up + R@neg@L@um)
        result = tmp_u[nex,1:-1,:,:]
    return U2w(result, γ)

def Lax(w, γ=1.4, C=0.5, t=100):
    N = w[:,0,0].size
    #print(N, w.shape, w.size)
    tmp_w = np.expand_dims(np.pad(w,((1,),(0,),(0,)), 'edge'), 0).repeat(2, axis=0)
    #print(tmp_w.shape, tmp_w.size)
    for n in range(t):
        cur = n%2
        nex = (n%2 + 1)%2
        #print("!")
        A = w2A(tmp_w[cur,:,:,:], γ)
        F = w2F(tmp_w[cur,:,:,:], γ)
        for i in range(N):
            I = i+1
            tmp_w[nex, I:I+1, :, :] =  tmp_w[cur, I:I+1, :, :]\
                    -0.5*C*(F[I+1:I+2,:,:] - F[I-1:I,:,:])\
                    +0.5*C*C*(0.5*(A[I+1:I+2,:,:] + A[I:I+1,:,:])@(F[I+1:I+2,:,:] - F[I:I+1,:,:])\
                    -0.5*(A[I:I+1,:,:] + A[I-1:I,:,:])@(F[I:I+1,:,:] - F[I-1:I,:,:]))
        result = tmp_w[nex,1:-1,:,:]
    return result

def minmod(a, b):
    return 0 if  a * b < 0 else min([a, b]) if b > 0 else max([a, b])

def limiter(x, C=1, t=1):
    N = x.size
    tmp = np.zeros((2, N), dtype=x.dtype)
    tmp[0] = x.copy()
    tmp[1] = x.copy()
    result = tmp[0]
    for n in range(t):
        cur = n%2
        nex = (n%2 + 1)%2
        for i in range(N):
            I = i - 2
            tmp[nex,I+1] =  tmp[cur,I+1] - C*(tmp[cur, I+1] - tmp[cur, I]) - 0.5 * C * (1 - C) *\
            ( minmod(tmp[cur, I+1]-tmp[cur, I], tmp[cur, I+2]-tmp[cur, I+1]) - \
                        minmod(tmp[cur, I]-tmp[cur, I-1], tmp[cur, I+1]-tmp[cur, I]) )
            result = tmp[nex]

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
    parser.add_argument("-t", "--times", default="0.5,0.75", type=str, help="time")
    args = parser.parse_args()

    Ts = [float(idx) for idx in args.times.split(',')]
    methods = args.methods.split(',')
    print(Ts)
    if args.numbers != 0:
        res = (args.end-args.start)/args.numbers
    else:
        res = args.resolution
    # Δx: args.resolution
    x = np.arange(args.start, args.end, res)
    # C is Δt/Δx
    C = args.ratio/4.7
    # Δt
    t = C * res

    w = init_w(x)
    fig, axs = plt.subplots(3*len(Ts),
                            len(methods),
                            figsize=(40, 12))
    for (T, i) in zip(Ts, range(len(Ts))):
        for (method, j) in zip(methods, range(len(methods))):
            n_t = int(T/t)

            γ = 1.4


            # simu output
            print("γ = " + str(γ) + ", C = " + str(C) + ", n_t = "+ str(n_t))
            if method == "Upwind":
                S1 = Upwind(w, γ, C, n_t)
            elif method == "LaxWendroff":
                S1 = Lax(w, γ, C, n_t)
            else:
                print("error input function")
            print('x: ', x.shape)
            print('w: ', w.shape)
            print('S1: ', S1[:,:,:].shape)
            ref = funcRef(x, C, T)
            axs[i*3 + 0][j].plot(x, S1[:, 0, 0])
            axs[i*3 + 0][j].plot(x, ref[0])
            axs[i*3 + 2][j].plot(x, S1[:, 1, 0])
            axs[i*3 + 2][j].plot(x, ref[2])
            axs[i*3 + 1][j].plot(x, S1[:, 2, 0])
            axs[i*3 + 1][j].plot(x, ref[1])
    plt.show()


