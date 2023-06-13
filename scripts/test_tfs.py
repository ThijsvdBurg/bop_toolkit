import math
import numpy as np
import bop_toolkit_lib.transform as transform
from scipy.spatial.transform import Rotation

# from transform import random_vector
# rdvec1 = random_vector(3)
# rdvec2 = random_vector(3)

def dummy_example():
    origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]

    # object relative to world
    roll_o_WF, pitch_o_WF, yaw_o_WF = 0.0, 0.0, -math.pi*.25
    x_o_WF, y_o_WF, z_o_WF = 10.0, 0.0, .3
    Rx_o_WF = transform.rotation_matrix(roll_o_WF, xaxis)
    Ry_o_WF = transform.rotation_matrix(pitch_o_WF, yaxis)
    Rz_o_WF = transform.rotation_matrix(yaw_o_WF, zaxis)
    R_o_WF = transform.concatenate_matrices(Rx_o_WF, Ry_o_WF, Rz_o_WF)
    R_o_WF[0,3] = x_o_WF
    R_o_WF[1,3] = y_o_WF
    R_o_WF[2,3] = z_o_WF





    # robot relative to world
    roll_r_WF, pitch_r_WF, yaw_r_WF = 0.0, 0.0, 0.0
    x_r_WF, y_r_WF, z_r_WF = 0.0, 0.0, 0.5
    Rx_r_WF = transform.rotation_matrix(roll_r_WF, xaxis)
    Ry_r_WF = transform.rotation_matrix(pitch_r_WF, yaxis)
    Rz_r_WF = transform.rotation_matrix(yaw_r_WF, zaxis)
    R_r_WF = transform.concatenate_matrices(Rx_r_WF, Ry_r_WF, Rz_r_WF)
    R_r_WF[0,3] = x_r_WF
    R_r_WF[1,3] = y_r_WF
    R_r_WF[2,3] = z_r_WF

    # camera relative to robot
    roll_c_RF, pitch_c_RF, yaw_c_RF = 0.0, (24.2/360*math.pi*2), 0.0
    x_c_RF, y_c_RF, z_c_RF = 0.0, 0.0, 2.5
    Rx_c_RF = transform.rotation_matrix(roll_c_RF, xaxis)
    Ry_c_RF = transform.rotation_matrix(pitch_c_RF, yaxis)
    Rz_c_RF = transform.rotation_matrix(yaw_c_RF, zaxis)
    R_c_RF = transform.concatenate_matrices(Rx_c_RF, Ry_c_RF, Rz_c_RF)
    R_c_RF[0,3] = x_c_RF
    R_c_RF[1,3] = y_c_RF
    R_c_RF[2,3] = z_c_RF

    # corner 1 relative to object
    roll_c1_OF, pitch_c1_OF, yaw_c1_OF = 0.0, 0.0, 0.0
    x_c1_OF, y_c1_OF, z_c1_OF = 0.25, 0.25, -0.3
    Rx_c1_OF = transform.rotation_matrix(roll_c1_OF, xaxis)
    Ry_c1_OF = transform.rotation_matrix(pitch_c1_OF, yaxis)
    Rz_c1_OF = transform.rotation_matrix(yaw_c1_OF, zaxis)
    R_c1_OF = transform.concatenate_matrices(Rx_c1_OF, Ry_c1_OF, Rz_c1_OF)
    R_c1_OF[0,3] = x_c1_OF
    R_c1_OF[1,3] = y_c1_OF
    R_c1_OF[2,3] = z_c1_OF


    T_c_WF = np.matmul(R_c_RF, R_r_WF)

    # T_o_CF = np.matmul(transform.inverse_matrix(np.matmul(R_c_RF, R_r_WF)))


    print(T_c_WF)




# print(yaw)
def object_CF():
    # R_o_WF_sci = 
    T_W_R = np.eye(4)
    R = Rotation.from_euler("XYZ",[0,0,0], degrees=True).as_matrix()
    T_W_R[:3,:3] = R
    T_W_R[:3,3 ] = np.array([0,0,-.5])
    T_R_W = np.linalg.inv(T_W_R)

    T_R_C = np.eye(4)
    R = Rotation.from_euler("XYZ",[0,0,0], degrees=True).as_matrix()
    T_R_C[:3,:3] = R
    T_R_C[:3,3 ] = np.array([0,0,.5])
    T_C_R = np.linalg.inv(T_R_C)

    T_W_O = np.eye(4)
    R = Rotation.from_euler("XYZ",[0,-10,180], degrees=True).as_matrix()
    T_W_O[:3,:3] = R
    T_W_O[:3,3 ] = np.array([10,0,.25])
    T_O_W = np.linalg.inv(T_W_O)

    # T_C_W = origin_c_in_a = T_C_R@T_R_W@np.array([0,0,0,1]).reshape(4,1)

    # T_W_C = np.matmul(T_W_R, T_R_C)
    # T_C_W = np.linalg.inv(T_W_C)
    # T_C_O = np.matmul(T_C_W, T_W_O)
    T_R_O = np.matmul(T_R_W, T_W_O)
    T_C_O = np.matmul(T_C_R, T_R_O)
    T_C_O_alt1 = np.matmul(T_C_R, np.matmul(T_R_W, T_W_O))
    T_C_O_alt2 = np.matmul(np.linalg.inv(T_R_C), np.matmul(np.linalg.inv(T_W_R), T_W_O))
    
    T_O_O1 = np.eye(4)
    T_O_O1[0,3] = .25
    T_O_O1[1,3] = 0.0
    T_O_O1[2,3] = -.25


    T_R_01 = np.matmul( T_R_W, np.matmul(T_W_O, T_O_O1) )
    T_C_O1     = np.matmul( T_C_R,  T_R_01)
    print('T_C_O1\n',T_C_O1)
    T_C_O1_alt1     = np.matmul( T_C_R,  np.matmul( T_R_W, np.matmul(T_W_O, T_O_O1) ))
    print('T_C_O1_alt1\n',T_C_O1_alt1)
    T_C_O1_alt2     = np.matmul( np.linalg.inv(T_R_C),  np.matmul( np.linalg.inv(T_W_R), np.matmul(T_W_O, T_O_O1) ))
    print('T_C_O1_alt2\n',T_C_O1_alt2)

    # print('T_C_O1\n',T_C_O1)


    # print('T_R_W\n',T_W_R)
    # print('T_W_C\n',T_W_C)
    # print('T_C_W\n',T_W_C)
    # print('T_C_O\n',T_C_O)
    # print('T_C_O1\n',T_C_O1)

    return T_C_O

def main():
    print('starting...')

    object_CF()

    return

if __name__ == main():
    main()