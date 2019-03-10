import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv


''' Part I: initialization
'''

with open('data/data.pickle', 'rb') as f:
    data = pickle.load(f)
    
t = data['t']  # timestamps [s]

x_init  = data['x_init'] # initial x position [m]
y_init  = data['y_init'] # initial y position [m]
th_init = data['th_init'] # initial theta position [rad]

# input signal
v  = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]

# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]


v_var = 1  # translation velocity variance: Initial is 0.01
om_var = 5  # rotational velocity variance: Initial is 0.01
r_var = 0.1  # range measurements variance: Initial is 0.1
b_var = 0.1  # bearing measurement variance: Initial is 0.1

Q_km = np.diag([v_var, om_var]) # input noise covariance 
cov_y = np.diag([r_var, b_var])  # measurement noise covariance 

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init]) # initial state
P_est[0] = np.diag([1, 1, 0.1]) # initial state covariance


''' Part II: angle wrap-up function
'''

# Wraps angle to (-pi,pi] range
def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x


''' Part III: measurement update function 
'''

def measurement_update(lk, rk, bk, P_check, x_check):
    # 1. Compute measurement Jacobian
    ## 1.1 A data type change & preprocess
    dis = d[0]
    x_check[2] = wraptopi(x_check[2])
    bk = wraptopi(bk)
    
    ## 1.2 Compute partial derivatives using the package sympy,
    ##     The source code is within the script, "symbolic_solve.py"
    ph1_x1 =  (dis*np.cos(x_check[2]) - lk[0] + x_check[0])/np.sqrt((-dis*np.sin(x_check[2]) + lk[1] - x_check[1])**2 + (-dis*np.cos(x_check[2]) + lk[0] - x_check[0])**2)
    ph1_x2 =  (dis*np.sin(x_check[2]) - lk[1] + x_check[1])/np.sqrt((-dis*np.sin(x_check[2]) + lk[1] - x_check[1])**2 + (-dis*np.cos(x_check[2]) + lk[0] - x_check[0])**2)
    ph1_x3 =  (-dis*(-dis*np.sin(x_check[2]) + lk[1] - x_check[1])*np.cos(x_check[2]) + dis*(-dis*np.cos(x_check[2]) + lk[0] - x_check[0])*np.sin(x_check[2]))/np.sqrt((-dis*np.sin(x_check[2]) + lk[1] - x_check[1])**2 + (-dis*np.cos(x_check[2]) + lk[0] - x_check[0])**2)
    
    ph2_x1 =  -(dis*np.sin(x_check[2]) - lk[1] + x_check[1])/((-dis*np.sin(x_check[2]) + lk[1] - x_check[1])**2 + (-dis*np.cos(x_check[2]) + lk[0] - x_check[0])**2)
    ph2_x2 =  -(-dis*np.cos(x_check[2]) + lk[0] - x_check[0])/((-dis*np.sin(x_check[2]) + lk[1] - x_check[1])**2 + (-dis*np.cos(x_check[2]) + lk[0] - x_check[0])**2)
    ph2_x3 =  dis*(dis*np.sin(x_check[2]) - lk[1] + x_check[1])*np.sin(x_check[2])/((-dis*np.sin(x_check[2]) + lk[1] - x_check[1])**2 + (-dis*np.cos(x_check[2]) + lk[0] - x_check[0])**2) - dis*(-dis*np.cos(x_check[2]) + lk[0] - x_check[0])*np.cos(x_check[2])/((-dis*np.sin(x_check[2]) + lk[1] - x_check[1])**2 + (-dis*np.cos(x_check[2]) + lk[0] - x_check[0])**2) - 1

    
    
    H_k = np.array([[ph1_x1, ph1_x2, ph1_x3], [ph2_x1, ph2_x2, ph2_x3]])
    
    M_k = np.eye(2)
    
    # 2. Compute Kalman Gain
    R_k = cov_y # measurement noise
    
    ###### Debug: Check size of each array #######
    
    # print(P_check.shape)
    # print(H_k.shape)
    # print(H_k)
    # print(ph1_x1)
    # print(ph2_x1)
    # print(M_k.shape)
    # print(R_k.shape)
    # print(K_k.shape)
    # print(H_k.shape)
    # print(P_check.shape)
    
    ###### End of Debug ##########################
    
    K_k = P_check @ H_k.T @ inv(H_k @ P_check @ H_k.T + M_k @ R_k @ M_k.T)
    
    
    # 3. Correct predicted state (remember to wrap the angles to [-pi,pi])
    
    ## 3-1: compute y_check
    y_check_1 = ((lk[0] - x_check[0] - dis * np.cos(x_check[2])) ** 2 + (lk[1] - x_check[1] - dis * np.sin(x_check[2])) ** 2) ** (0.5)
    y_check_2 = np.arctan2(lk[1] - x_check[1] - dis * np.sin(x_check[2]), lk[0] - x_check[0] - dis * np.cos(x_check[2])) - x_check[2]
    
    # -------- Wrap-up check point 1: --------- #
    y_check_2 = wraptopi(y_check_2) 
    # ----------------------------------------- #
    
    y_check = np.array([y_check_1, y_check_2]).T # n has a mean equal to [0, 0].T, so nvm
    
    ## 3-2: compute y_l_k
    y_l_k = np.array([rk, bk]).T
    y_l_k[1] = wraptopi(y_l_k[1])
    
    ## 3-3: compute x_hat
    x_hat = x_check + K_k @ (y_l_k - y_check)
    
    # -------- Wrap-up check point 2: --------- #
    x_hat[2] = wraptopi(x_hat[2])
    # ----------------------------------------- #
    
    # 4. Correct covariance
    
    ## 4-1 computer P_hat
    P_hat = (np.eye(3) - K_k @ H_k) @ P_check
    
    #return x_check, P_check
    return x_hat, P_hat






''' Part IV: Main loop
'''

#### 5. Main Filter Loop #######################################################################
for k in range(1, len(t)):  # start at 1 because we've set the initial prediciton

    delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)
    
    # -------- Wrap-up check point 3: --------- #
    x_est[k-1, 2] = wraptopi(x_est[k-1, 2])
    # ----------------------------------------- #
    
    # 1. Update state with odometry readings (remember to wrap the angles to [-pi,pi])
    x_check = np.zeros(3)
    u_k_minus_1 = np.array([v[k-1], om[k-1]]).T
    
    # 2. Motion model jacobian with respect to last state
    F_km = np.zeros([3, 3])
    F_km = np.eye(3)
    F_km[0, 2] = -np.sin(x_est[k-1, 2]) * v[k] * delta_t # initially it's v[k] # v[k-1] passed the test
    F_km[1, 2] = np.cos(x_est[k-1, 2]) * v[k] * delta_t # initially it's om[k] # v[k-1] passed the test
    
    
    # 3. Motion model jacobian with respect to noise
    L_km = np.zeros([3, 2])
    L_km[0, 0] = np.cos(x_est[k-1, 2]) * delta_t
    L_km[1, 0] = np.sin(x_est[k-1, 2]) * delta_t
    L_km[2, 1] = 1 * delta_t
    
    ## Supplement of # 1.
    x_check = x_est[k-1] + delta_t * L_km @ u_k_minus_1
    
    
    # -------- Wrap-up check point 4: --------- #
    x_check[2] = wraptopi(x_check[2])
    # ----------------------------------------- #
    
    # 4. Propagate uncertainty
    P_check = F_km @ P_est[k-1, :, :] @ F_km.T + L_km @ Q_km @ L_km.T
    
    # 5. Update state estimate using available landmark measurements
    for i in range(len(r[k])):
        x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check)

    # Set final state predictions for timestep
    x_est[k, 0] = x_check[0]
    x_est[k, 1] = x_check[1]
    x_est[k, 2] = x_check[2]
    P_est[k, :, :] = P_check


''' Part V: Plotting
'''

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(x_est[:, 0], x_est[:, 1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.show()

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t[:], x_est[:, 2])
ax.set_xlabel('Time [s]')
ax.set_ylabel('theta [rad]')
ax.set_title('Estimated trajectory')
plt.show()