#!/usr/bin/env python3
from __future__ import annotations
import math, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'stix',
    'font.size':8.5,'axes.labelsize':9,'axes.linewidth':0.75,'legend.fontsize':7.8,
    'xtick.labelsize':8,'ytick.labelsize':8,'xtick.direction':'in','ytick.direction':'in',
    'lines.linewidth':1.0,'lines.markersize':4.2,'legend.frameon':False,
})
BLUE='#0072BD'; RED='#A2142F'; ORANGE='#D95319'; BLACK='#000000'

# ---------- mesh + FD ----------
def rounded_candidate(x, B, objective):
    cands=sorted(set([max(1,min(B-1,int(math.floor(x)))),max(1,min(B-1,int(math.ceil(x))))]))
    return min(cands,key=lambda n:(objective(n),abs(n-x)))

def split_symmetric(muL,muR,kL,kR,B): return B//2, B-B//2

def split_max(muL,muR,kL,kR,B):
    x=B*math.sqrt(muL)/(math.sqrt(muL)+math.sqrt(muR))
    def obj(n): return max(muL/n**2,muR/(B-n)**2)
    n=rounded_candidate(x,B,obj); return n,B-n

def split_bal(muL,muR,kL,kR,B):
    AL=muL**2*kL; AR=muR**2*kR
    x=B*AL**(1/3)/(AL**(1/3)+AR**(1/3))
    def obj(n): return AL/n**2+AR/(B-n)**2
    n=rounded_candidate(x,B,obj); return n,B-n

def resolving_floor(N,sigma=2.0,zeta=0.7):
    return int(math.ceil(sigma*math.log(N)/zeta))

def constrained_split_max(muL,muR,kL,kR,B,N,zeta=0.7,sigma=2.0):
    m=resolving_floor(N,sigma,zeta)
    if 2*m>=B: m=max(1,B//4)
    x=B*math.sqrt(muL)/(math.sqrt(muL)+math.sqrt(muR))
    x=min(max(x,m),B-m)
    def obj(n): return max(muL/n**2,muR/(B-n)**2)
    cands=sorted(set([max(m,min(B-m,int(math.floor(x)))),max(m,min(B-m,int(math.ceil(x))))]))
    n=min(cands,key=lambda q:(obj(q),abs(q-x))); return n,B-n

def constrained_split_bal(muL,muR,kL,kR,B,N,zeta=0.7,sigma=2.0):
    m=resolving_floor(N,sigma,zeta)
    if 2*m>=B: m=max(1,B//4)
    AL=muL**2*kL; AR=muR**2*kR
    x=B*AL**(1/3)/(AL**(1/3)+AR**(1/3))
    x=min(max(x,m),B-m)
    def obj(n): return AL/n**2+AR/(B-n)**2
    cands=sorted(set([max(m,min(B-m,int(math.floor(x)))),max(m,min(B-m,int(math.ceil(x))))]))
    n=min(cands,key=lambda q:(obj(q),abs(q-x))); return n,B-n

def make_mesh_custom(N,eps,muL,muR,kL=1.0,kR=1.0,sigma=2.0,rule='max'):
    assert N%4==0
    NM=N//2; B=N-NM
    tauL=min(0.25,sigma*eps*math.log(N)/kL)
    tauR=min(0.25,sigma*eps*math.log(N)/kR)
    if rule=='sym': NL,NR=split_symmetric(muL,muR,kL,kR,B)
    elif rule=='max': NL,NR=constrained_split_max(muL,muR,kL,kR,B,N)
    elif rule=='bal': NL,NR=constrained_split_bal(muL,muR,kL,kR,B,N)
    else: raise ValueError(rule)
    left=np.linspace(0,tauL,NL+1)
    middle=np.linspace(tauL,1-tauR,NM+1)[1:]
    right=np.linspace(1-tauR,1,NR+1)[1:]
    return np.concatenate([left,middle,right]),NL,NM,NR,tauL,tauR

def make_mesh_given_split(N,eps,NL,kL=1.0,kR=1.0,sigma=2.0):
    NM=N//2; B=N-NM; NR=B-NL
    tauL=min(0.25,sigma*eps*math.log(N)/kL)
    tauR=min(0.25,sigma*eps*math.log(N)/kR)
    left=np.linspace(0,tauL,NL+1); middle=np.linspace(tauL,1-tauR,NM+1)[1:]; right=np.linspace(1-tauR,1,NR+1)[1:]
    return np.concatenate([left,middle,right]),NL,NM,NR,tauL,tauR

def thomas(lower,diag,upper,rhs):
    a=np.array(lower,float); d=np.array(diag,float); c=np.array(upper,float); b=np.array(rhs,float)
    for i in range(1,len(d)):
        w=a[i-1]/d[i-1]; d[i]-=w*c[i-1]; b[i]-=w*b[i-1]
    x=np.empty(len(d)); x[-1]=b[-1]/d[-1]
    for i in range(len(d)-2,-1,-1): x[i]=(b[i]-c[i]*x[i+1])/d[i]
    return x

def solve_fd(x,eps,f,alpha,gamma,bfun=None):
    if bfun is None: bfun=lambda y:np.ones_like(y)
    xi=x[1:-1]; hi=x[1:-1]-x[:-2]; hip=x[2:]-x[1:-1]; fac=2/(hi+hip)
    lo=-eps**2*fac/hi; up=-eps**2*fac/hip; diag=eps**2*fac*(1/hi+1/hip)+bfun(xi)
    rhs=np.asarray(f(xi),float).copy(); rhs[0]-=lo[0]*alpha; rhs[-1]-=up[-1]*gamma
    ui=thomas(lo[1:],diag,up[:-1],rhs)
    U=np.empty(len(x)); U[0]=alpha; U[-1]=gamma; U[1:-1]=ui; return U

# ---------- exact manufactured BVP, constant reaction coefficient ----------
def phiL(x,eps):
    den=-np.expm1(-2/eps); return (np.exp(-x/eps)-np.exp(-(2-x)/eps))/den

def phiR(x,eps):
    den=-np.expm1(-2/eps); return (np.exp(-(1-x)/eps)-np.exp(-(1+x)/eps))/den

def dphiL(x,eps):
    den=-np.expm1(-2/eps); return -(np.exp(-x/eps)+np.exp(-(2-x)/eps))/(eps*den)

def dphiR(x,eps):
    den=-np.expm1(-2/eps); return (np.exp(-(1-x)/eps)+np.exp(-(1+x)/eps))/(eps*den)

def uexact(x,eps,AL,AR): return 1+x+AL*phiL(x,eps)+AR*phiR(x,eps)
def duexact(x,eps,AL,AR): return 1+AL*dphiL(x,eps)+AR*dphiR(x,eps)

def solve_manufactured(N,eps,AL,AR,rule):
    x,NL,NM,NR,*_=make_mesh_custom(N,eps,abs(AL),abs(AR),1,1,rule=rule)
    U=solve_fd(x,eps,lambda y:1+y,1+AL,2+AR)
    return x,U,(NL,NM,NR)

gq,wq=np.polynomial.legendre.leggauss(16)
def error_metrics(x,U,eps,AL,AR):
    """Return nodal max, reconstructed L-infinity, balanced, L2, and balanced derivative errors."""
    nodal_max=float(np.max(np.abs(U-uexact(x,eps,AL,AR))))
    recon_max=nodal_max
    L2=0.0; H1=0.0
    for i in range(len(x)-1):
        a=x[i]; b=x[i+1]; h=b-a
        xx=(a+b)/2+(h/2)*gq; s=(xx-a)/h
        Uh=U[i]+s*(U[i+1]-U[i]); dUh=(U[i+1]-U[i])/h
        ev=Uh-uexact(xx,eps,AL,AR); ed=dUh-duexact(xx,eps,AL,AR)
        recon_max=max(recon_max,float(np.max(np.abs(ev))))
        L2 += (h/2)*float(np.dot(wq,ev*ev)); H1 += (h/2)*float(np.dot(wq,ed*ed))
    bal=math.sqrt(L2+eps*H1)
    return nodal_max,recon_max,bal,math.sqrt(L2),math.sqrt(eps*H1)

# ---------- exact frozen balanced functions ----------
def theta(z): return 2*math.asinh(z/2)
def frozen_F(z):
    th=theta(z)
    istar=1/z if abs(z-th)<1e-15 else math.log(z/th)/(z-th)
    cand={0,max(0,int(math.floor(istar))),max(0,int(math.ceil(istar)))}
    return max(math.exp(-th*i)-math.exp(-z*i) for i in cand)

def frozen_I(z):
    if z < 0.06:
        return z*z/24 - 43*z**4/11520 + 589*z**6/3870720
    th=theta(z); a=math.exp(-th); b=math.exp(-z)
    return 0.5 + (1-a)**2/(z*(1-a*a)) - 2*(1-a)*(1-b)/(z*(1-a*b))

def frozen_K(z):
    if z < 0.06:
        return 73*z**4/11520 - 4369*z**6/3870720 + 25321*z**8/132710400
    th=theta(z); a=math.exp(-th); b=math.exp(-z); d=1-a
    A=(1+a+a*a)/3
    J=(1-b)/z - d*(1-(1+z)*b)/z**2
    L=A/(1-a*a)+1/(2*z)-2*J/(1-a*b)
    return z*L

def frozen_bal_sq(mu,kappa,eps,z): return mu*mu*(kappa*frozen_I(z)+(eps/kappa)*frozen_K(z))

# ---------- parameter-uniform pilot ----------
def pilot_exact(x,eps):
    x=np.asarray(x,float)
    if abs(eps-1.0)<1e-14:
        p=-0.5*x*np.exp(x); p0=0.0; p1=-0.5*math.e
    else:
        p=np.exp(x)/(1-eps**2); p0=1/(1-eps**2); p1=math.e/(1-eps**2)
    return p-p0*phiL(x,eps)-p1*phiR(x,eps)

def pilot_error(N,eps,rule):
    x,NL,NM,NR,*_=make_mesh_custom(N,eps,1.0,math.e,1.0,1.0,rule=rule)
    U=solve_fd(x,eps,np.exp,0.0,0.0)
    return float(np.max(np.abs(U-pilot_exact(x,eps)))),(NL,NM,NR)

# ---------- canonical S-type / Bakhvalov-S extension ----------
def stype_phi(t,N,family='bakhvalov_s'):
    t=np.asarray(t,float)
    if family=='shishkin':
        return 2*t*math.log(N)
    if family=='bakhvalov_s':
        return -np.log(1-2*t*(1-1/N))
    raise ValueError(family)

def make_stype_mesh_given_split(N,eps,NL,kL=1.0,kR=1.0,sigma=2.0,family='bakhvalov_s'):
    NM=N//2; B=N-NM; NR=B-NL
    tauL=sigma*eps*math.log(N)/kL; tauR=sigma*eps*math.log(N)/kR
    if tauL>=0.25 or tauR>=0.25:
        raise ValueError('S-type comparison is restricted to the uncapped regime.')
    tl=np.arange(NL+1,dtype=float)/(2*NL)
    tr=np.arange(NR+1,dtype=float)/(2*NR)
    left=sigma*eps/kL*stype_phi(tl,N,family)
    rdist=sigma*eps/kR*stype_phi(tr,N,family)
    right=1-rdist[::-1]
    middle=np.linspace(tauL,1-tauR,NM+1)[1:-1]
    x=np.concatenate([left,middle,right])
    if len(x)!=N+1 or not np.all(np.diff(x)>0):
        raise RuntimeError('Invalid S-type mesh.')
    return x,NL,NM,NR,tauL,tauR

def pure_left_layer_metrics_stype(N,eps,NL,family='bakhvalov_s'):
    x,*_=make_stype_mesh_given_split(N,eps,NL,family=family)
    U=solve_fd(x,eps,lambda y:np.zeros_like(y),1.0,0.0)
    nodal=float(np.max(np.abs(U-phiL(x,eps))))
    L2=0.0; H1=0.0
    for i in range(len(x)-1):
        a=x[i]; b=x[i+1]; h=b-a
        xx=(a+b)/2+(h/2)*gq; s=(xx-a)/h
        Uh=U[i]+s*(U[i+1]-U[i]); dUh=(U[i+1]-U[i])/h
        ev=Uh-phiL(xx,eps); ed=dUh-dphiL(xx,eps)
        L2+=(h/2)*float(np.dot(wq,ev*ev)); H1+=(h/2)*float(np.dot(wq,ed*ed))
    return nodal,math.sqrt(L2+eps*H1)

def manufactured_metrics_stype(N,eps,ratio,NL,family='bakhvalov_s'):
    x,*_=make_stype_mesh_given_split(N,eps,NL,family=family)
    U=solve_fd(x,eps,lambda y:1+y,1+ratio,3.0)
    return error_metrics(x,U,eps,float(ratio),1.0)

def bakhvalov_balanced_interpolation_constant(N):
    a=1-1/N
    return math.sqrt(a**3*(1+1/N)/3)
