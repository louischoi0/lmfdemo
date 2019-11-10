"""
  먼저 로직 구현에 필요한 모듈들은 어떤 것들이 있는지 분석 후
  그에 맞게 클래스들을 구현, 전체 프로그램 구조들을 그려본다.
  이러한 작업은 프로그램 작성중 반복적으로 일어나야 하며
  전체 구조는 수시로 바뀔수 있다는 것을 염두해 두어
  확장성이 높고 변경이 쉽도록 모듈간 의존성을 최소화(디커플링)
  하는 방식으로 구현해야 한다.
"""

import numpy as np

class RangeUnexpectedError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self,message):
        self.message = message

class OptionUnexpectedError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self,message):
        self.message = message

class sparseMatrix :
  def __init__(self,xlen,ylen) :
    """
      Sparse Matrix는 최소한의 정보량을 가지고
      대부분 0으로 채워져 있는 행렬값을 표현하기 위한 구조
      self.datas Dictionary 데이터에 있는 값들만 2차원 좌표를 정수로
      변환해 저장,
      이에 채워져있는 idx에만 저장된 값을 리턴하고,
      그외의 인덱스에는 0을 반환한다.
    """
    self.xlen = xlen
    self.ylen = ylen
    self.datas = {}

  def reconv(self,v) :
    return ( v % self.xlen , v // self.xlen )

  def append(self,count,axis="x") :
    v = []

    newxlen =  self.xlen + count if axis == "x" else self.xlen

    if axis == "x" :

      for k in self.datas :
        x,y = self.reconv(k)

        newloc = ( newxlen * y ) + x

        v.append((newloc,self.datas[k]))

      self.datas = { x[0] : x[1] for x in v }

    if axis == "x" :
      self.xlen += count

    elif axis == "y" :
      self.ylen += count

    else :
      raise OptionUnexpectedError("Axis {} is not valid.".format(axis))
      
  def convloc(self,x,y) :
    """
      2차원 좌표를 정수로 변환하여
      딕셔너리의 한개의 키값으로만 접근하도록 구현
    """
    return (self.xlen * y) + x

  def setorinc(self,x,y) :
    locidx = self.convloc(x,y)

    try :
      self.datas[locidx] += 1

    except KeyError :
      self.setvalue(x,y,1)

  def setvalue(self,x,y,v) :
    if x >= self.xlen or y >= self.ylen :
      raise RangeUnexpectedError("({},{}) out of range. it`s shape is ({},{})".format(x,y,self.xlen,self.ylen))

    locidx = self.convloc(x,y)
    self.datas[locidx] = v

  def get(self,x,y) :
    if x >= self.xlen or y >= self.ylen :
      raise RangeUnexpectedError("({},{}) out of range. it`s shape is ({},{})".format(x,y,self.xlen,self.ylen))

    try :
      locidx = self.convloc(x,y)
      return self.datas[locidx]

    except KeyError :
      return 0

  def remove(self,x,y) :
    locidx = self.convloc(x,y)
    del self.datas[locidx]

  def show(self) :
    """
      Sparse Matrix는 데이터들의 부분만 메모리에 보관해 있으므로
      사람이 눈으로 확인하기 매우 힘들다.
      이러한 특징으로 인해 show 함수와 같이 쉽게 데이터를 확인하는
      메소드를 구현하면 개발/테스트시 매우 큰 도움이 된다.
    """
    p = ""

    for y in range(0,self.ylen) :
      for x in range(0,self.xlen) :
        p += str(self.get(x,y))
        p += " "

      p += "\n"

    print(p)

  def tondarray(self) :
    res = np.zeros((self.xlen,self.ylen)) 

    for k in self.datas :
      x,y = self.reconv(k)
      res[y][x] = self.datas[k]

    return res

  @staticmethod
  def fromndarray(ndarr) :
    r = sparseMatrix(ndarr.shape[0],ndarr.shape[1])

    idx = 0
    ndarr = ndarr.flatten()    

    for v in ndarr :

      if v != 0 :
        r.datas[idx] = v

      idx += 1

    return r

if __name__ == "__main__" :
    m = sparseMatrix(5,5)
    m.setvalue(1,1,9)
    m.setvalue(2,2,9)
    m.setvalue(1,2,9)
    m.setvalue(3,3,9)
    m.show()
    #m.append(3,axis="z")

    nd = m.tondarray()
    r = sparseMatrix.fromndarray(nd)
    r.show() 

    print(r)

    try :
      m.setvalue(100,100,10)
    except RangeUnexpectedError :
      print("")