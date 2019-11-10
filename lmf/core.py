import pandas as pd
import numpy as np
from math import exp
import math
from lmf.collections.sparsematrix import sparseMatrix 

np.random.seed(0)

import sys

class lmf :

  class optimizer :
    def __init__(self) :
      self.mti = None
      self.gti = None
      self.init = False
      self.method = None

    def prepare(self,shape) :
      self.gti = np.zeros(shape)

    def RMSProp(self,x,uidx,grad,fudge_factor=1e-6,stepsize=1e-2) :
      self.gti[uidx] = stepsize * self.gti[uidx] 
      self.gti[uidx] += (1 - stepsize) * (grad ** 2)

      adjusted_grad = grad / ( fudge_factor + np.sqrt(self.gti[uidx]) )
      x[uidx] -= stepsize * adjusted_grad 

    def adaGrad(self,x,uidx,grad,fudge_factor=1e-6,stepsize=1e-2) :
      self.gti[uidx] += grad ** 2

      adjusted_grad = grad / ( fudge_factor + np.sqrt(self.gti[uidx]) )
      x[uidx] -= stepsize * adjusted_grad 

    def momentum(self,x,uidx,grad,stepsize=1e-2,forget_rate=1e-7) :
      self.gti[uidx] *= 0.9
      self.gti[uidx] += grad * 0.9 * forget_rate

      adjusted_grad = grad + self.gti[uidx]

      x[uidx] -= stepsize * adjusted_grad 

  def optimizer_factory(self,method,**kwargs) :
    opt = self.optimizer()
    opt.method = getattr(opt,method)
    return opt

  def __init__(self,interactions,method="adaGrad") :
    self.interactions = interactions
    self.history = []
    self.optz = self.optimizer_factory(method)

  def kernel(self,intreactions,usercount,itemcount) :
    rmtx = sparseMatrix(usercount,itemcount)
    self.usercount = usercount
    self.itemcount = itemcount

    for _,v in intreactions.iterrows() :
      u = v["user"]
      i = v["item"]

      rmtx.setorinc(u,i)

    return rmtx

  def init(self,interactions,factordm,uc,ic) :
    self.rmtx = self.kernel(interactions,uc,ic)
    self.tmtx = np.zeros((self.usercount,self.itemcount))

    """
      0 ~ 1 사이의 값으로 랜덤하게 초기화
      factordm 은 하나의 벡터가 가질 Feature 개수.
    """
    self.userfactor = np.random.rand(self.usercount,factordm)
    self.itemfactor = np.random.rand(self.itemcount,factordm)

    self.userbiases = np.random.rand(self.usercount)
    self.itembiases = np.random.rand(self.itemcount)

  def assume(self) :
    for uidx,uf in enumerate(self.userfactor) :
      for iidx,itf in enumerate(self.itemfactor) :
        userbias = self.userbiases[uidx]
        itembias = self.itembiases[iidx]

        self.tmtx[uidx,iidx] = lmf.p(uf,itf,userbias,itembias)

  def gradfn(self,uidx,lambdax=0.5,alpha=0.3) :
    dx = 0
    for iidx,itf in enumerate(self.itemfactor) :

      r = self.rmtx.get(uidx,iidx)

      ub = self.userbiases[uidx]
      ib = self.itembiases[iidx]
      
      x = self.userfactor[uidx]
      y = itf

      h0 = exp(np.dot(x,y) * ub + ib)
      
      dx += (r*alpha*y) + ( ( (y * (1 + r*alpha) ) * h0 ) / (h0 + 1) ) - (lambdax*x)
    
    return dx

  def _optimize(self,iterations,**kwargs) :
    gti = np.zeros_like(self.userfactor)
    from copy import deepcopy
    w = deepcopy(self.userfactor)

    for it in range(iterations) :
      for uidx,_ in enumerate(self.userfactor) :
        grad = self.gradfn(uidx)
        self.optz.method(w,uidx,grad,**kwargs)

      self.userfactor = w  
      self.assume()
      _loss = self.lossobj()
      self.history.append(_loss)

    return w

  def optimizeuserfactor(self,iterations=10,stepsize=1e-6,**kwargs) :
    self.optz.prepare(self.userfactor.shape)

    w = self._optimize(iterations,**kwargs)
    
    self.userfactor = w
    self.assume()
    return self.lossobj()

  def lossobj(self,alpha=0.1,scalinglambda=0.5) :
    """
      Iteration 안에서 빈번히 호출될때 Sub function을 활용하면 매우 좋다.
    """
    def _obj(uv,iv,ub,ib) :

        h0 = np.dot(uv,iv) + ub + ib

        h1 = alpha * r * h0
        h2 = (1 + (alpha * r)) * math.log(1 + exp(h0))

        h3 = -0.5 * scalinglambda * (np.linalg.norm(uv) ** 2) + (-0.5  * scalinglambda * (np.linalg.norm(iv) ** 2))
        # 아래는 논문 4페이지의 수식 3 참조. 
        return h1 - h2 + h3

    loss = 0

    for u in range(self.usercount) :
      for i in range(self.itemcount) :
        r = self.rmtx.get(u,i) 

        uv = self.userfactor[u]
        iv = self.itemfactor[i]

        ub = self.userbiases[u]
        ib = self.itembiases[i]

        loss += _obj(uv,iv,ub,ib)

    return loss

  def loss(self,a=0.9) :
    """
      a는 튜닝 파라미터
      
    """
    c = 1
    
    for u in range(self.usercount) :
      for i in range(self.itemcount) :
        p = self.tmtx[u,i]
        ar = self.rmtx.get(u,i) * a

        f = p^ar

        assert f < 1

        # 아래의 값은 논문 3페이지의 수식 1,2 참조.
        c *= f*(1-f) 

    return c

  @staticmethod
  def p(x,y,userbias,itembias) :
    a = exp(np.dot(x,y) + userbias + itembias)
    return a / (1 + a)
