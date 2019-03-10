from numpy import *
from numpy.linalg import inv
import numpy as np


def sph_to_cart(epsilon, alpha, r):
  """
  Transform sensor readings to Cartesian coordinates in the sensor frames. 
  """
  p = np.zeros(3)  # Position vector 
  
  # Your code here
  p[0] = r * np.cos(epsilon) * np.cos(alpha)
  p[1] = r * np.cos(epsilon) * np.sin(alpha)
  p[2] = r * np.sin(epsilon)
  
  return p
  
def estimate_params(P):
  """
  Estimate parameters from sensor readings in the Cartesian frame.
  Each row in the P matrix contains a single measurement.
  """
  param_est = np.zeros(3)
  
  # Your code here
  row = P.shape[0]
  A = np.zeros((row, 3))
  b = np.zeros(row)
  for i in range(0, row):
    p_cart = sph_to_cart(P[i, 0], P[i, 1], P[i, 2])
    A[i, 0] = 1
    A[i, 1] = p_cart[0]
    A[i, 2] = p_cart[1]
    b[i]    = p_cart[2]
  
  # try to be consistant
  # x_hat = inv(A.T @ A) @ A.T @ b
  x_hat = np.matmul(np.matmul(inv(np.matmul(A.T, A)), A.T), b)
  
  # print(x_hat.shape)

  param_est[0] = x_hat[0]
  param_est[1] = x_hat[1]
  param_est[2] = x_hat[2]

  return param_est

if __name__ == '__main__':
  inp = np.array([[1, 2, 5], [1.1, 2.1, 5.2], [1.2, 2.2, 5.4], [1.3, 2.3, 5.6]])
  param_est_result = estimate_params(inp)
  print(param_est_result)

# (3.92, 0.69, 0.35)
