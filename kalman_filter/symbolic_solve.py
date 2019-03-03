from sympy import *
import math

xl = Symbol('lk[0]')
xk = Symbol('x_check[0]')
yl = Symbol('lk[1]')
yk = Symbol('x_check[1]')
thetak = Symbol('x_check[2]')
d = Symbol('dis')

y1 = sqrt((xl - xk - d * cos(thetak)) ** 2 + (yl - yk - d * sin(thetak))** 2)
y2 = atan2(yl - yk - d * sin(thetak), xl - xk - d * cos(thetak)) - thetak

py1_pxk = diff(y1, xk, 1)
py1_pyk = diff(y1, yk, 1)
py1_pthetak = diff(y1, thetak, 1)

py2_pxk = diff(y2, xk, 1)
py2_pyk = diff(y2, yk, 1)
py2_pthetak = diff(y2, thetak, 1)

print("ph1_x1 = ", py1_pxk)
print("ph1_x2 = ", py1_pyk)
print("ph1_x3 = ", py1_pthetak)

print("ph2_x1 = ", py2_pxk)
print("ph2_x2 = ", py2_pyk)
print("ph2_x3 = ", py2_pthetak)
