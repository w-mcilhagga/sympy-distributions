#!/usr/bin/env python
# coding: utf-8

import sympy as sp
from sympy.functions.combinatorial.factorials import binomial, factorial
from sympy.functions.special.gamma_functions import gamma


class RandomVar:
    pass

# in all of these classes, the name is the name given to a random variable with
# that distribution. For example, if we say x ~ X, then the name should be 'x'

class Binomial(sp.Symbol, RandomVar):
    def __new__(cls, name, p=sp.Symbol('p'), n=sp.Symbol('n')):
        bv = sp.Symbol.__new__(cls,name,positive=True)
        p._assumptions['positive']=True
        n._assumptions['positive']=True
        bv.p = p
        bv.n = n
        return bv
    
class Geometric(sp.Symbol, RandomVar):
    def __new__(cls, name, p=sp.Symbol('p')):
        gv = sp.Symbol.__new__(cls,name,positive=True)
        gv.p = p
        return gv

class Gaussian(sp.Symbol, RandomVar):
    def __new__(cls, name, mu=sp.Symbol('mu'), sigma=sp.Symbol('sigma')):
        gv = sp.Symbol.__new__(cls,name)
        gv.mu = mu
        gv.sigma = sigma
        return gv
    
class Poisson(sp.Symbol, RandomVar):
    def __new__(cls, name, _lambda=sp.Symbol('lambda')):
        # use _lambda instead of lambda
        pv = sp.Symbol.__new__(cls,name,positive=True)
        pv._lambda = _lambda
        return pv
    
class NegativeBinomial(sp.Symbol, RandomVar):
    def __new__(cls, name, r=sp.Symbol('r'), p=sp.Symbol('p')):
        nbv = sp.Symbol.__new__(cls,name,positive=True)
        nbv.r = r
        nbv.p = p
        return nbv
    
class Uniform(sp.Symbol, RandomVar):
    def __new__(cls, name, a=sp.Symbol('a'), b=sp.Symbol('b')):
        uv = sp.Symbol.__new__(cls,name)
        uv.a = a
        uv.b = b
        return uv

class ChiSquared(sp.Symbol, RandomVar):
    def __new__(cls, name, k=sp.Symbol('k')):
        cv = sp.Symbol.__new__(cls,name,positive=True)
        cv.k = k
        return cv

class Gamma(sp.Symbol, RandomVar):
    # parameterized by k, theta
    def __new__(cls, name, k=sp.Symbol('k'), theta=sp.Symbol('theta')):
        gv = sp.Symbol.__new__(cls,name,positive=True)
        gv.k = k
        gv.theta = theta
        return gv

def Gamma_alt(name):
    # alternative formulation parameterized by alpha=k, beta=1/theta
    return Gamma(name, k=sp.Symbol('alpha'), theta=1/sp.Symbol('beta'))

def Exponential(name, _lambda=sp.Symbol('lambda')):
    return Gamma(name, k=1, theta=1/_lambda,positive=True)

class InverseGaussian(sp.Symbol, RandomVar):
    def __new__(cls, name, mu=sp.Symbol('mu'), _lambda=sp.Symbol('lambda')):
        igv = sp.Symbol.__new__(cls,name,positive=True)
        igv.mu = mu
        igv._lambda = _lambda
        return igv

# MGFs need to be defined because sympy can't integrate some of the PDFs

MGF = {
    # 1st arg in mgf is the rv, second is the dummy variable t
    Binomial: lambda y, t: ((1-y.p) +y.p*sp.exp(t))**y.n,
    Gaussian: lambda y, t: sp.exp(y.mu*t+y.sigma**2*t**2/2),
    Poisson: lambda y, t: sp.exp(y._lambda* (sp.exp(t)-1)),
    NegativeBinomial: lambda y, t: ((1-y.p)/(1-y.p*sp.exp(t)))**y.r,
    Geometric: lambda y, t: y.p*sp.exp(t)/(1-(1-y.p)*sp.exp(t)),
    Uniform: lambda y, t: (sp.exp(t*y.b)-sp.exp(t*y.a))/(t*(y.b-y.a)),
    ChiSquared: lambda y, t: (1-2*t)**(-y.k/2),
    Gamma: lambda y, t: (1-t*y.theta)**(-y.k),
    InverseGaussian: lambda y, t: sp.exp(y._lambda/y.mu*(1-sp.sqrt(1-2*y.mu**2*t/y._lambda)))
}

PDF = {
    Binomial: lambda y: binomial(y,y.n)*y.p**y*(1-y.p)**(y.n-y),
    Gaussian: lambda y: sp.exp(-(y-y.mu)**2/(2*y.sigma**2))/sp.sqrt(2*sp.pi*y.sigma**2),
    Poisson: lambda y: y._lambda**y*sp.exp(-y._lambda)/factorial(y),
    NegativeBinomial: lambda y: binomial(y+y.r-1,y)*(1-y.p)**y.r*y.p**y,
    Geometric: lambda y: (1-y.p)*(y-1)*y.p,
    Uniform: lambda y: 1/(y.b-y.a),
    Gamma: lambda y: y**(y.k-1)*sp.exp(-y/y.theta)/(y.theta**y*gamma(y)),
    InverseGaussian: lambda y: sp.sqrt(y._lambda/(2*sp.pi*y**3))*sp.exp(-y._lambda*(y-y.mu)**2/(2*y.mu**2*y))
}

def E(expr):
    # expected value of an expression
    t = sp.Symbol('t')
    
    def moment(rv,n):
        t = sp.Symbol('t')
        dmdt = sp.diff(MGF[rv.__class__](rv,t),t,n)
        m = dmdt.subs(t,0)
        if m != sp.nan:
            return m
        m = sp.limit(dmdt, t, 0)
        return m
    
    def expected(expr):
        cls = expr.__class__
        if cls == sp.Add:
            return sp.Add(*[expected(elem) for elem in expr.args])
        if cls == sp.Mul:
            return sp.Mul(*[expected(elem) for elem in expr.args])
        if cls == sp.Pow and isinstance(expr.args[0],RandomVar):
            if not expr.args[1].is_integer:
                raise Exception("Can't do expected values of non-integer powers")                
            if expr.args[1]<0:
                raise Exception("Can't do expected values of negative powers (including division)")
            return moment(expr.args[0], expr.args[1])
        if isinstance(expr, RandomVar):
            return moment(expr, 1)
        return expr
    
    return sp.factor(expected(sp.expand(expr)))

def pr(rv):
    return PDF[rv.__class__](rv)

def log(expr):
    # a log function which forces expansion
    return sp.log(expr).expand(force=True)





