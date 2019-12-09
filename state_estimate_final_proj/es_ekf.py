# Starter code for the Coursera SDC Course 2 final project.
#
# Author: Trevor Ablett and Jonathan Kelly
# University of Toronto Institute for Aerospace Studies
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion
from numpy.linalg import inv

#### 1. Data ###################################################################################

################################################################################################
# This is where you will load the data from the pickle files. For parts 1 and 2, you will use
# p1_data.pkl. For Part 3, you will use pt3_data.pkl.
################################################################################################
with open('data/pt1_data.pkl', 'rb') as file:
    data = pickle.load(file)

################################################################################################
# Each element of the data dictionary is stored as an item from the data dictionary, which we
# will store in local variables, described by the following:
#   gt: Data object containing ground truth. with the following fields:
#     a: Acceleration of the vehicle, in the inertial frame
#     v: Velocity of the vehicle, in the inertial frame
#     p: Position of the vehicle, in the inertial frame
#     alpha: Rotational acceleration of the vehicle, in the inertial frame
#     w: Rotational velocity of the vehicle, in the inertial frame
#     r: Rotational position of the vehicle, in Euler (XYZ) angles in the inertial frame
#     _t: Timestamp in ms.
#   imu_f: StampedData object with the imu specific force data (given in vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   imu_w: StampedData object with the imu rotational velocity (given in the vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   gnss: StampedData object with the GNSS data.
#     data: The actual data
#     t: Timestamps in ms.
#   lidar: StampedData object with the LIDAR data (positions only).
#     data: The actual data
#     t: Timestamps in ms.
################################################################################################
gt = data['gt']
imu_f = data['imu_f']
imu_w = data['imu_w']
gnss = data['gnss']
lidar = data['lidar']

################################################################################################
# Let's plot the ground truth trajectory to see what it looks like. When you're testing your
# code later, feel free to comment this out.
################################################################################################
gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Ground Truth trajectory')
ax.set_zlim(-1, 5)
plt.show()

################################################################################################
# Remember that our LIDAR data is actually just a set of positions estimated from a separate
# scan-matching system, so we can insert it into our solver as another position measurement,
# just as we do for GNSS. However, the LIDAR frame is not the same as the frame shared by the
# IMU and the GNSS. To remedy this, we transform the LIDAR data to the IMU frame using our 
# known extrinsic calibration rotation matrix C_li and translation vector t_i_li.
#
# THIS IS THE CODE YOU WILL MODIFY FOR PART 2 OF THE ASSIGNMENT.
################################################################################################
# Correct calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.1).
C_li = np.array([
   [ 0.99376, -0.09722,  0.05466],
   [ 0.09971,  0.99401, -0.04475],
   [-0.04998,  0.04992,  0.9975 ]
])

# Incorrect calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.05).
# C_li = np.array([
#      [ 0.9975 , -0.04742,  0.05235],
#      [ 0.04992,  0.99763, -0.04742],
#      [-0.04998,  0.04992,  0.9975 ]
# ])

# translation vector
t_i_li = np.array([0.5, 0.1, 0.5])

# Transform from the LIDAR frame to the vehicle (IMU) frame.
lidar.data = (C_li @ lidar.data.T).T + t_i_li

#### 2. Constants ##############################################################################

################################################################################################
# Now that our data is set up, we can start getting things ready for our solver. One of the
# most important aspects of a filter is setting the estimated sensor variances correctly.
# We set the values here.
################################################################################################
var_imu_f = 0.10
var_imu_w = 0.25
var_gnss  = 0.01
var_lidar = 1.00

################################################################################################
# We can also set up some constants that won't change for any iteration of our solver.
################################################################################################
g = np.array([0, 0, -9.81])  # gravity
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian

#### 3. Initial Values #########################################################################

################################################################################################
# Let's set up some initial values for our ES-EKF solver.
################################################################################################
p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions
p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  # covariance matrices at each timestep

# Set initial values.
p_est[0] = gt.p[0]
v_est[0] = gt.v[0]
q_est[0] = Quaternion( euler = gt.r[0] ).to_numpy()
p_cov[0] = np.zeros(9)  # covariance of estimate
gnss_i  = 0
lidar_i = 0

#### 4. Measurement Update #####################################################################

################################################################################################
# Since we'll need a measurement update for both the GNSS and the LIDAR data, let's make
# a function for it.
################################################################################################
def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    
    # construct H_k = [I, 0, 0] (size = 3 x 10)
    H_k = np.zeros(3, 10)
    H_k[0][0] = 1
    H_k[1][1] = 1
    H_k[2][2] = 1

    # 3.1 Compute Kalman Gain
    # evaluate size chain: (10 x 10) x (10 x 3) x ( (3 x 10) x (10 x 10) x (10 x 3) + (3 x 3) )
    # K_k should have a size: (10 x 3)
    K_k = p_cov_check @ H_k.T @ inv(H_k @ p_cov_check @ H_k.T + sensor_var)

    # 3.2 Compute error state
    # evaluate size chain: (10 x 3) x ( (3 x 1) - (3 x 1) )
    # delta_x_k should have a size: (10 x 1)
    delta_x_k = K_k @ (y_k - p_check)

    # 3.3 Correct predicted state
    p_hat = p_check + delta_x_k[0:2]
    v_hat = v_check + delta_x_k[3:5]
    # q_hat = ??? weird algebraic calculation # TODO: understand how to process q

    # 3.4 Compute corrected covariance
    # evaluate size chain: ( (10 x 10) - (10 x 3) x (3 x 10) ) x (10 x 10)
    p_cov_hat = ( np.identity(10) - K_k @ H_k ) @ p_cov_check

    return p_hat, v_hat, q_hat, p_cov_hat

#### 5. Main Filter Loop #######################################################################

################################################################################################
# Now that everything is set up, we can start taking in the sensor data and creating estimates
# for our state in a loop.
################################################################################################

# Define some suppotive variables
R_GNSS  =  np.identity(3) * var_gnss    # covariance matrix related to GNSS
R_Lidar =  np.identity(3) * var_lidar   # covariance matrix related to Lidar
t_imu   =  imu_f.t                      # timestanps of imu
t_gnss  =  gnss.t                       # timestamps of gnss
t_lidar =  lidar.t                      # timestamps of lidar 
F_k     =  np.identity(10)
L_k     =  np.zeros(10, 6)
L_k[3:8, :] =  np.identity(6)
Q       =  np.identity(6)               # covariance matrix related to noise of IMU
Q[0:2, 0:2] = Q[0:2, 0:2] * var_imu_f   # covariance matrix related to special force of IMU
Q[3:5, 3:5] = Q[3:5, 3:5] * var_imu_w   # covariance matrix related to rotational speed of IMU

for k in range(1, imu_f.data.shape[0]):  # start at 1 b/c we have initial prediction from gt
    delta_t = imu_f.t[k] - imu_f.t[k - 1]

    Q_k = Q * delta_t * delta_t

    # 1. Update state with IMU inputs
    ## 1-1: update position states
    p_est[k] = p_est[k-1] + delta_t * v_est[k-1] + delta_t ** 2 / 2 * (C_li @ imu_f.data[k-1] + g) # TODO: obtain f_k_1 (done, f_k_1 = imu_f.data[k-1]); confirm C_ns == C_li 
    ## 1-2: update velocity states
    v_est[k] = v_est[k-1] + delta_t * (C_li @ imu_f.data[k-1] + g)                                 # TODO: obtain f_k_1 (done, f_k_1 = imu_f.data[k-1]); confirm C_ns == C_li

    ## 1-3: update orientation states
    
    ### construct theta increment
    ### theta_icm: theta increament
    ### size of theta_icm: (3 x 1)
    theta_icm      = imu_w.data[k-1] * delta_t
    theta_icm_norm = np.sqrt( theta_icm[0] ** 2 + theta_icm[1] ** 2 + theta_icm[2] ** 2 )
    
    ### construct quaternion based on theta increment
    q_w = np.sin(theta_icm_norm / 2)
    q_v = theta_icm / theta_icm_norm * np.cos(theta_icm_norm / 2)

    ### construct quaternion process matrix based on the quaternion info
    Omega = np.identity(4) * q_w 
    Omega[0, 1:3]   = -q_v.T
    Omega[1:3, 0]   = q_v
    Omega[1:3, 0]   = q_v
    q_R             = np.zeros(3, 3)
    q_R[0, 1]       = q_v[2]
    q_R[1, 0]       = -q_v[2]
    q_R[0, 2]       = -q_v[1]
    q_R[2, 0]       = q_v[1]
    q_R[1, 2]       = q_v[0]
    q_R[2, 1]       = -q_v[0]
    Omega[1:3, 1:3] = q_R

    ### finally, process quaternion state update
    q_est[k] = Omega @ q_est[k-1]

    # 1.1 Linearize the motion model and compute Jacobians
    F_k[0:2, 3:5] = np.identity(3) * delta_t
    # F_k[3:5, 6:8] = ??? * delta_t # TODO: understand how to process q
    # where ??? = [C_{ns} f_{k-1}]_x, fill it out later

    # 2. Propagate uncertainty
    p_cov = F_k @ p_cov @ F_k.T + L_k @ Q_k @ L_k.T

    # 3. Check availability of GNSS and LIDAR measurements
    ## at t_imu[k] timestamp GNSS has input
    if np.any( t_gnss == t_imu[k] ):
        [ p_est[k], v_est[k], q_est[k], p_cov[k] ] = measurement_update( R_GNSS, p_cov[k], gnss.data[k], p_est[k], v_est[k], q_est[k] )
    
    ## at t_imu[k] timestamp Lidar has input
    if np.any( t_lidar == t_imu[k] ):
        [ p_est[k], v_est[k], q_est[k], p_cov[k] ] = measurement_update( R_Lidar, p_cov[k], gnss.data[k], p_est[k], v_est[k], q_est[k] )

    # Update states (save) # TODO: make sure no extra processes needed

#### 6. Results and Analysis ###################################################################

################################################################################################
# Now that we have state estimates for all of our sensor data, let's plot the results. This plot
# will show the ground truth and the estimated trajectories on the same plot. Notice that the
# estimated trajectory continues past the ground truth. This is because we will be evaluating
# your estimated poses from the part of the trajectory where you don't have ground truth!
################################################################################################
est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_zlabel('Up [m]')
ax.set_title('Ground Truth and Estimated Trajectory')
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_zlim(-2, 2)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_yticks([0, 50, 100, 150, 200])
ax.set_zticks([-2, -1, 0, 1, 2])
ax.legend(loc=(0.62,0.77))
ax.view_init(elev=45, azim=-50)
plt.show()

################################################################################################
# We can also plot the error for each of the 6 DOF, with estimates for our uncertainty
# included. The error estimates are in blue, and the uncertainty bounds are red and dashed.
# The uncertainty bounds are +/- 3 standard deviations based on our uncertainty (covariance).
################################################################################################
error_fig, ax = plt.subplots(2, 3)
error_fig.suptitle('Error Plots')
num_gt = gt.p.shape[0]
p_est_euler = []
p_cov_euler_std = []

# Convert estimated quaternions to euler angles
for i in range(len(q_est)):
    qc = Quaternion(*q_est[i, :])
    p_est_euler.append(qc.to_euler())

    # First-order approximation of RPY covariance
    J = rpy_jacobian_axis_angle(qc.to_axis_angle())
    p_cov_euler_std.append(np.sqrt(np.diagonal(J @ p_cov[i, 6:, 6:] @ J.T)))

p_est_euler = np.array(p_est_euler)
p_cov_euler_std = np.array(p_cov_euler_std)

# Get uncertainty estimates from P matrix
p_cov_std = np.sqrt(np.diagonal(p_cov[:, :6, :6], axis1=1, axis2=2))

titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
for i in range(3):
    ax[0, i].plot(range(num_gt), gt.p[:, i] - p_est[:num_gt, i])
    ax[0, i].plot(range(num_gt),  3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].plot(range(num_gt), -3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].set_title(titles[i])
ax[0,0].set_ylabel('Meters')

for i in range(3):
    ax[1, i].plot(range(num_gt), \
        angle_normalize(gt.r[:, i] - p_est_euler[:num_gt, i]))
    ax[1, i].plot(range(num_gt),  3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].plot(range(num_gt), -3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].set_title(titles[i+3])
ax[1,0].set_ylabel('Radians')
plt.show()

#### 7. Submission #############################################################################

################################################################################################
# Now we can prepare your results for submission to the Coursera platform. Uncomment the
# corresponding lines to prepare a file that will save your position estimates in a format
# that corresponds to what we're expecting on Coursera.
################################################################################################

# Pt. 1 submission
p1_indices = [9000, 9400, 9800, 10200, 10600]
p1_str = ''
for val in p1_indices:
    for i in range(3):
        p1_str += '%.3f ' % (p_est[val, i])
with open('pt1_submission.txt', 'w') as file:
    file.write(p1_str)

# Pt. 2 submission
# p2_indices = [9000, 9400, 9800, 10200, 10600]
# p2_str = ''
# for val in p2_indices:
#     for i in range(3):
#         p2_str += '%.3f ' % (p_est[val, i])
# with open('pt2_submission.txt', 'w') as file:
#     file.write(p2_str)

# Pt. 3 submission
# p3_indices = [6800, 7600, 8400, 9200, 10000]
# p3_str = ''
# for val in p3_indices:
#     for i in range(3):
#         p3_str += '%.3f ' % (p_est[val, i])
# with open('pt3_submission.txt', 'w') as file:
#     file.write(p3_str)