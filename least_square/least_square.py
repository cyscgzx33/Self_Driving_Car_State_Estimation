import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Store the voltage and current data as column vectors.
I = np.mat([0.2, 0.3, 0.4, 0.5, 0.6]).T
V = np.mat([1.23, 1.38, 2.06, 2.47, 3.17]).T

plt.scatter(np.asarray(I), np.asarray(V))

plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.show()

# Define the H matrix, what does it contain?
H = I

# Now estimate the resistance parameter.
R = inv(H.T * H) * H.T * V

print('The slope parameter (i.e., resistance) for the best-fit line is:')
print(R)

I_line = np.arange(0, 0.8, 0.1)
I_line = np.mat([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
V_line = R*I_line

plt.scatter(np.asarray(I), np.asarray(V))
plt.plot(I_line, V_line)
plt.xlabel('current (A)')
plt.ylabel('voltage (V)')
plt.grid(True)
plt.show()
