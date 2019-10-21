"""
  article : https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf
  blog : https://louischoi0.github.io
"""

import pandas as pd
import numpy as np
from math import exp
import math
from sparsematrix import sparseMatrix

np.random.seed(0)

class lmf :
  def __init__(self,interactions) :
    self.interactions = interactions

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
    """
      factor들로 연산한 값과 rmtx 행렬원소에 대한 오차의 총합
    """

    for uidx,uf in enumerate(self.userfactor) :
      for iidx,itf in enumerate(self.itemfactor) :
        userbias = self.userbiases[uidx]
        itembias = self.itembiases[iidx]

        self.tmtx[uidx,iidx] = lmf.p(uf,itf,userbias,itembias)

  def gradfn(self,uidx,lambdax=0.5,alpha=0.1) :
    dx = 0
    for iidx,itf in enumerate(self.itemfactor) :

      r = self.rmtx.get(uidx,iidx)

      ub = self.userbiases[uidx]
      ib = self.itembiases[iidx]
      
      x = self.userfactor[uidx]
      y = itf

      h0 = exp(np.dot(x,y) * ub + ib)
      
      dx += ( (y * (1 + r*alpha) ) * h0 ) / (h0 + 1) - (lambdax*x)
    
    """
      논문 4페이지의 수식 5 참조.
    """
    return dx

  def _optimize(self,iterations=10,fudge_factor=1e-6,stepsize=1e-2) :
    gti = np.zeros_like(self.userfactor)
    from copy import deepcopy

    w = deepcopy(self.userfactor)

    for it in range(iterations) :
      for uidx,_ in enumerate(self.userfactor) :
        """
          한 유저 벡터에 대해 최적화 할 gradient 함수를 받아 최적화 합니다.
          Adagrad 알고리즘을 최대한 단순화 하여 수식화 하였습니다.

          이 알고리즘의 기본적인 아이디어는 ‘지금까지 많이 변화하지 않은 변수들은 step size를 크게 하고, 
          지금까지 많이 변화했던 변수들은 step size를 작게 하자’ 라는 것이다. 
          자주 등장하거나 변화를 많이 한 변수들의 경우 optimum에 가까이 있을 확률이 높기 때문에 작은 크기로 이동하면서 세밀한 값을 조정하고, 
          적게 변화한 변수들은 optimum 값에 도달하기 위해서는 많이 이동해야할 확률이 높기 때문에 먼저 
          빠르게 loss 값을 줄이는 방향으로 이동하는 방식이다.

          출처 : http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html
        """

        grad = self.gradfn(uidx)

        """
          gti 변수에 각 유저 벡터 원소들에 대해 움직인 강도를 반복 합산.
        """
        gti[uidx] += grad ** 2
        
        """
          gti 값을 통해 학습률 조정.
        """
        adjusted_grad = grad / ( fudge_factor + np.sqrt(gti[uidx]) )
        w[uidx] -= stepsize * adjusted_grad 

    return w

  def optimizeuserfactor(self,iterations=10,fudge_factor=1e-6,stepsize=1e-2) :
    w = self._optimize(iterations,fudge_factor,stepsize)
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
        """
          scalinglambda 는 데이터가 스케일 아웃 될때 보정해주는 수치이다.
          크게 신경쓰지 말고 h1,h2의 값 위주로 논문 수식을 확인하여 보자.  
        """
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

    """
      아래의 스칼라 값을 최대화 시키는 것이 목적이다.
      a * b * c * d 의 값을 최대화 시키는 것은 결국
      ln(a) + ln(b) + ln(c) + ln(d) 의 값을 최대화 시키는 것과 같다.
      loss함수의 식과 _obj안의 식을 비교하여 보자.
    """

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

        """
          f는 확률 값이므로 언제나 0보다 크고 1보다 작아야 한다.
          assert는 데이터의 정합성을 검사할 때 사용되는데,
          만약 무조건 지켜져야 하는 조건이 있다면
          assert를 이용해 데이터를 검증한다.
          만약 조건을 벗어나면 AssertionError를 띄우며 프로그램이 죽는다.
        """
        assert f < 1

        # 아래의 값은 논문 3페이지의 수식 1,2 참조.
        c *= f*(1-f) 

    return c

  @staticmethod
  def p(x,y,userbias,itembias) :
    a = exp(np.dot(x,y) + userbias + itembias)
    return a / (1 + a)

if __name__ == "__main__" :

  ti = np.random.randint(5,size=(20,2))
  #df = pd.read_csv("aa.csv")

  df = pd.DataFrame(ti,columns=["user","item"])
  
  user_count = max(df["user"].values)
  item_count = max(df["item"].values)

  l = lmf(df)

  l.init(df,5,user_count+1,item_count+1)

  loss = l.lossobj()

  print("R행렬은 아래와 같습니다.")
  l.rmtx.show()

  print("Initial loss value : {}".format(loss))
  print()

  l.assume()
  print("초기 추측값은 아래와 같습니다.")
  print(l.tmtx)
  print()

  loss_after = l.optimizeuserfactor(1200)

  print("After Optimize loss value : {}".format(loss_after))
  print()
  print("최적화 후 추측값은 아래와 같습니다.")
  print(l.tmtx)

  from sys import exit
  exit(0)
