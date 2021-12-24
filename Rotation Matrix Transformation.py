import numpy as np
import cv2

def findRotMat(alpha, beta, gamma):
    cos_alpha = np.cos(np.radians(alpha))
    sin_alpha = np.sin(np.radians(alpha))
    cos_beta = np.cos(np.radians(beta))
    sin_beta = np.sin(np.radians(beta))
    cos_gamma = np.cos(np.radians(gamma))
    sin_gamma = np.sin(np.radians(gamma))
    yaw_alpha = np.array(((cos_alpha, sin_alpha, 0), (-sin_alpha, cos_alpha, 0), (0, 0 ,1))) #z
    roll_beta = np.array(((1, 0, 0), (0, cos_beta, sin_beta), (0, -sin_beta, cos_beta))) #x
    yaw_gamma = np.array(((cos_gamma, sin_gamma, 0), (-sin_gamma, cos_gamma, 0), (0, 0 ,1))) #z

    matmul1 = np.matmul(roll_beta, yaw_gamma)
    rotMat1 = np.matmul(yaw_alpha, matmul1)
    rotMat2 = np.linalg.inv(rotMat1)
    return rotMat1, rotMat2

if __name__ == "__main__":
    alpha = 45
    beta = 30
    gamma = 60
    rotMat1, rotMat2 = findRotMat(alpha, beta, gamma)
    print(rotMat1)
    print(rotMat2)