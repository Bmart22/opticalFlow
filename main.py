"""
Sources:
Face landmark detection in Python:
https://medium.com/analytics-vidhya/facial-landmarks-and-face-detection-in-python-with-opencv-73979391f30e
"""

import sys
import cv2 as cv
import numpy as np


# Add newest frame to list that contains the three most recent frames
def update_frames(local_frames, new_frame):
    local_frames[0] = local_frames[1]
    local_frames[1] = local_frames[2]
    local_frames[2] = new_frame
    return


# Calculate the time-dimension sobel filter
def time_sobel(local_frames):
    # Init output array
    dim = local_frames[0].shape
    filter_output = np.zeros( dim )
    
    # Loop over all pixels
    for r in range(dim[0]):
        for c in range(dim[1]):
            filter_output = local_frames[0] - local_frames[1]
        
    return


def horn_schunk(prev_u, prev_v, x_sobel, y_sobel, t_sobel, num_iter):
    # Calculate the average velocity within local areas
    # Use gaussian blurring to average
    ave_u = cv.GaussianBlur(prev_u, (5, 5), 0, 0)
    ave_v = cv.GaussianBlur(prev_v, (5, 5), 0, 0)
    
    lamb = 0.5 # importance of the smoothness term
    #error_cutoff =
    iter = 0
    
    '''while (E > error_cutoff and iter < num_iter) :
        # Calculate the Optical Flow Constraint (total velocity is divided between x, y, t dimensions)
        P = x_sobel*ave_u + y_sobel*ave_v + t_sobel
        # Calculate the smoothing term (encourages velocity field to be smooth)
        D = lamb*lamb + x_sobel*x_sobel + y_sobel*y_sobel
        # take ratio
        P_D = P/D
        
        # Calculate the velocity of the current frame
        curr_u = prev_u - x_sobel * P_D
        curr_v = prev_v - y_sobel * P_D
        
        # Calculate velocity field error term (how close are we to optimal solution?)
        new_P = x_sobel*curr_u + y_sobel*curr_v + t_sobel
        new_D = lamb*lamb * (curr_u*curr_u + curr_v*curr_v)
        E = (new_P*new_P + new_D).sum()
        
        iter += 1'''
        
    # Filter out weak optical flow
    
    return


def main(argv):

    cap = cv.VideoCapture('./videos/IMG_7895.MOV')

    # parameters:
    ksize = 3 # sobel kernel size
    ddepth = -1 #cv2.CV_64F # sobel return type

    local_frames = [0, 0, 0]

    # Quickly find faces using Haar Cascades
    detector = cv.CascadeClassifier("haarcascade_frontalface_alt2.xml")

    # Find face landmarks
    face_points = []
    landmark_detector = cv.face.createFacemarkLBF()
    lbf_model = "lbfmodel.yaml"
    landmark_detector.loadModel(lbf_model)

    while True:
        # captures frame if stream not paused
        ret, frame = cap.read()

        # ret checks for correct frame read
        if not ret:
            print("Error: frame not read correctly")
            break

        frame = cv.flip(frame, 0)

        update_frames(local_frames, frame)

        # Get the x, y, and t derivatives
        grey_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        grad_x = cv.Sobel(grey_img, ddepth, 1, 0, ksize)
        grad_y = cv.Sobel(grey_img, ddepth, 0, 1, ksize)

        # Quickly find faces in a frame
        faces = detector.detectMultiScale(grey_img)

        # Detect landmarks on "grey_img"
        _, landmarks = landmark_detector.fit(grey_img, faces)

        for landmark in landmarks:
            for x, y in landmark[0]:
                # display landmarks on "grey_img" with white colour in BGR and thickness 1
                cv.circle(grey_img, (int(x), int(y)), 5, (255, 255, 255), 3)

        # key commands
        key = cv.waitKey(1)

        # quits if user presses q
        if key == ord('q'):
            break

        cv.imshow("Video", grey_img)

    # end video stream
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
