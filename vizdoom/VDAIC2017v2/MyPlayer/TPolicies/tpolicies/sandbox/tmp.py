from collections import namedtuple


class NetConfig(object):
  x = 3
  y = 4
  z = 6
  a = 7
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      self.__dict__[k] = v


cc = {
  'x': 99,
  'y': 42
}

nc = NetConfig()
nc2 = NetConfig(x=99, y=42)
nc3 = NetConfig(z=66, a=88)
print(nc)
print(nc2)