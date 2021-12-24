###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates. 
#                                            True if the intrinsic parameters are invariable.
#It is ok to add other functions if you need
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

def calibrate(imgname):

    threeD_coordinates = np.array([[[0,40,10]],
                        [[0,30,10]],
                        [[0,20,10]],
                        [[0,10,10]],
                        [[0,0,10]],
                        [[10,0,10]],
                        [[20,0,10]],
                        [[30,0,10]],
                        [[40,0,10]],
                        [[0,40,20]],
                        [[0,30,20]],
                        [[0,20,20]],
                        [[0,10,20]],
                        [[0,0,20]],
                        [[10,0,20]],
                        [[20,0,20]],
                        [[30,0,20]],
                        [[40,0,20]],
                        [[0,40,30]],
                        [[0,30,30]],
                        [[0,20,30]],
                        [[0,10,30]],
                        [[0,0,30]],
                        [[10,0,30]],
                        [[20,0,30]],
                        [[30,0,30]],
                        [[40,0,30]],
                        [[0,40,40]],
                        [[0,30,40]],
                        [[0,20,40]],
                        [[0,10,40]],
                        [[0,0,40]],
                        [[10,0,40]],
                        [[20,0,40]],
                        [[30,0,40]],
                        [[40,0,40]]])
    

    imagecoord = []
    original_realcoord = []

    img = imread(imgname)
    grayscale = cvtColor(img, COLOR_BGR2GRAY)

    criteria = (TERM_CRITERIA_EPS+TERM_CRITERIA_MAX_ITER, 30, 0.001)
    retval, corners = findChessboardCorners(grayscale, (4,9), None)
    if retval==True:
        cornersubpixels = cornerSubPix(grayscale, corners, (5,5), (-1,-1), criteria)
        img = drawChessboardCorners(grayscale, (4,9), cornersubpixels, retval)
        original_realcoord.append(threeD_coordinates)
        imagecoord.append(cornersubpixels)
    
    
    def intrinsicparams(realworldcoord, imageworldcoord):
        Augmentedmat = np.zeros((72, 12))
        for i in range(72):
            if i%2==0:
                Augmentedmat[i][0] = realworldcoord[0][i//2][0][0]
                Augmentedmat[i][1] = realworldcoord[0][i//2][0][1]
                Augmentedmat[i][2] = realworldcoord[0][i//2][0][2]
                Augmentedmat[i][3] = 1
                Augmentedmat[i][4] = 0
                Augmentedmat[i][5] = 0
                Augmentedmat[i][6] = 0
                Augmentedmat[i][7] = 0
                Augmentedmat[i][8] = -imageworldcoord[0][i//2][0][0]*realworldcoord[0][i//2][0][0]
                Augmentedmat[i][9] = -imageworldcoord[0][i//2][0][0]*realworldcoord[0][i//2][0][1]
                Augmentedmat[i][10] = -imageworldcoord[0][i//2][0][0]*realworldcoord[0][i//2][0][2]
                Augmentedmat[i][11] = -imageworldcoord[0][i//2][0][0]
            else:
                Augmentedmat[i][0] = 0
                Augmentedmat[i][1] = 0
                Augmentedmat[i][2] = 0
                Augmentedmat[i][3] = 0
                Augmentedmat[i][4] = realworldcoord[0][(i-1)//2][0][0]
                Augmentedmat[i][5] = realworldcoord[0][(i-1)//2][0][1]
                Augmentedmat[i][6] = realworldcoord[0][(i-1)//2][0][2]
                Augmentedmat[i][7] = 1
                Augmentedmat[i][8] = -imageworldcoord[0][(i-1)//2][0][1]*realworldcoord[0][(i-1)//2][0][0]
                Augmentedmat[i][9] = -imageworldcoord[0][(i-1)//2][0][1]*realworldcoord[0][(i-1)//2][0][1]
                Augmentedmat[i][10] = -imageworldcoord[0][(i-1)//2][0][1]*realworldcoord[0][(i-1)//2][0][2]
                Augmentedmat[i][11] = -imageworldcoord[0][(i-1)//2][0][1]

        u, s, VT = np.linalg.svd(Augmentedmat)
        VTrans = VT[11]
        #To reshape Vtrans into (3,4) matrix
        Xvals = np.reshape(VTrans,(3,4))
        x_one_one = Xvals[2][0]
        y_one_one = Xvals[2][1]
        z_one_one = Xvals[2][2]
        lambdagreek = (1/np.sqrt((x_one_one**2)+(y_one_one**2)+(z_one_one**2)))

        #now to get the m matrix its the mutiplication of lambda and augmented matrix
        M = lambdagreek*Xvals

        M_one = np.array([M[0][0],M[0][1],M[0][2]])
        M_two = np.array([M[1][0],M[1][1],M[1][2]])
        M_three = np.array([M[2][0],M[2][1],M[2][2]])
        M_one_trans = M_one.T
        M_two_trans = M_two.T
        M_three_trans = M_three.T
        O_x = np.matmul(M_one_trans,M_three)
        O_y = np.matmul(M_two_trans,M_three)
        f_x = np.sqrt((np.matmul(M_one_trans,M_one))-(O_x**2))
        f_y = np.sqrt((np.matmul(M_two_trans,M_two))-(O_y**2))

        intrinsic_params = np.array([[f_x, 0, O_x],[0, f_y, O_y],[0, 0, 1]])
        return intrinsic_params
    
    if __name__ == "__main__":
        intrinsic_params = intrinsicparams(original_realcoord, imagecoord)
        #as per the scaling factor multiplied to world coordinates, it does not change the intrinsci parameters
        is_constant = True
        return intrinsic_params, is_constant


if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)