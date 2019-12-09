# State Estimation and Localization for Self-Driving Cars <br />

## Run
* Simply `python3` to execute the code, preferably using `conda` environment

## Algorithm Implementation
### Error-State Kalman Filter for State Estimation and Localization
* Check the course's [Final Project](state_estimate_final_proj/), abd [src code](state_estimate_final_proj/es_ekf.py)
* Using *IMU* and *GNSS* data as measurement input data
* Using **error-state kinematics model** as the motion/measurement model
* Using **quaternion kinematics** as the rotation process update
* Implement the algorithm in `Python3`

## Reference
### Websites
* [Coursera website for this course: State Estimation and Locaization](https://www.coursera.org/learn/state-estimation-localization-self-driving-cars/home/welcome)
* [Coursera website for the other courses from University of Toronto about Self-driving Car topics](https://www.coursera.org/specializations/self-driving-cars)
### Papers & Textbooks
* [Quaternion kinematics for the error-state Kalman filter, Sola, 2017](https://arxiv.org/pdf/1711.02508.pdf)
* [Quaternions and Rotations, Jia, 2013](http://graphics.stanford.edu/courses/cs348a-17-winter/Papers/quaternion.pdf)