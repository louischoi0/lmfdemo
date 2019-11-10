from lmf.core import lmf
import numpy as np
import pandas as pd

from lmf.collections import sparsematrix
import sys

"""
    무도 정법 그알 뉴스 런닝맨 호텔 동백
a    [13,10,0,1,2,0,0]
b    [7,4,9,1,10,0,1]
c    [0,0,9,9,1,1,1]
d    [1,0,9,8,1,0,1]
e    [1,1,0,0,0,0,1]
f    [1,0,2,3,0,1,1]
g    [0,0,1,1,0,0,0]
i    [9,8,1,1,5,2,2]

"""

if __name__ == "__main__" :
  import sys
  print(sys.argv)

  ti = np.random.randint(5,size=(20,2))

  r = np.array([
    [3,0,0,1,4,0,0],
    [0,1,4,5,1,0,1],
    [1,2,1,1,1,1,1],
    [1,0,4,5,1,0,1],
    [1,1,0,0,0,0,1],
    [2,2,2,2,0,1,1],
    [0,0,1,1,0,0,0],
    [4,3,1,1,5,2,2]])

  rtx = sparsematrix.sparseMatrix.fromndarray(r)
  rtx.xlen = 7
  rtx.ylen = 8

  rtx.show()

  df = pd.DataFrame(ti,columns=["user","item"])
  
  user_count = max(df["user"].values)
  item_count = max(df["item"].values)

  #l = lmf(df)
  l = lmf(df,"adaGrad")

  l.init(df,5,7,7)
  l.rmtx = rtx

  loss = l.lossobj()

  print("R행렬은 아래와 같습니다.")
  l.rmtx.show()

  print("Initial loss value : {}".format(loss))
  print()

  l.assume()
  print("초기 추측값은 아래와 같습니다.")
  print(l.tmtx)
  print()
  iternum = int(sys.argv[1])

  from copy import deepcopy
  l2 = deepcopy(l)
  l3 = deepcopy(l)

  #loss_after = l2.optimizeuserfactor(o,iternum,forget_rate=1e-8,stepsize=1e-2)
  #loss_after2 = l.optimizeuserfactor(o1,iternum,fudge_factor=1e-3,stepsize=1e-2)

  loss_after = l.optimizeuserfactor(iternum,fudge_factor=1e-3,stepsize=1e-7)

  #loss_after_max = max(l.history)
  #loss_after_max1 = max(l2.history)
  loss_after_max = max(l.history)

  #print("After Optimize loss value : {} {} {}".format(loss_after,loss_after2,loss_after3))
  print("After Optimize loss value : {} ".format(loss_after))
  #print("After Optimize loss max value : {} {} {}".format(loss_after_max,loss_after_max1,loss_after_max2))
  print()
  print("최적화 후 추측값은 아래와 같습니다.")
  print(l.tmtx)

  import matplotlib.pyplot as plt
  xaxis = list( x for x in range(iternum) )
  plt.plot(xaxis,l.history)
  #plt.plot(xaxis,l.history,xaxis,l2.history,xaxis,l3.history,'r-')
  plt.show()
  from sys import exit
