import ctypes
from ctypes import *
pDll = windll.LoadLibrary("c2python.dll")
readPath = c_char_p(b"D:/Y0227/1x/Y0227_MC_20200601100254_01_01.jpg")
savePath = c_char_p(b"D:/photo/Dst.jpg")
pDll.c2p.restype = c_float
reslut = pDll.c2p(readPath,savePath)
print('覆盖率为：%.2f%%'%reslut)


