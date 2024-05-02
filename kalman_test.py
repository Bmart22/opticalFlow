#
# Brendan Martin, Phil Butler
# kalman_test.py
# CS 7180 Advanced Perception - Fall 2022 - 11/6/22

"""
Sources:
Face landmark detection in Python:
https://medium.com/analytics-vidhya/facial-landmarks-and-face-detection-in-python-with-opencv-73979391f30e
"""

import sys
import cv2 as cv
import numpy as np

# Time step frame-to-frame
delta_t = 1/60
#delta_t = 1


# Add newest facial features to list that contains the three most recent time steps
def update_features(local_features, new_feature):
    local_features[0] = local_features[1]
    local_features[1] = local_features[2]
    local_features[2] = new_feature
    return
    
# Calculate the position, velocity, and acceleration for each facial feature given three time steps
def format_state(local_features):
    x_state = np.zeros( (3, len(local_features[2])) )
    y_state = np.zeros( (3, len(local_features[2])) )
    
    for i in range(len(local_features[2])):
        # Copy position
        x_state[0,i] = local_features[2][i][0]
        y_state[0,i] = local_features[2][i][1]
        
        # Copy velocity
        x_state[1,i] = (local_features[2][i][0] - local_features[1][i][0]) / delta_t
        y_state[1,i] = (local_features[2][i][1] - local_features[1][i][1]) / delta_t
        
        # Copy acceleration
        v = (local_features[2][i][0] - local_features[1][i][0]) / delta_t
        u = (local_features[1][i][0] - local_features[0][i][0]) / delta_t
        x_state[2,i] = (v - u) / delta_t
        
        v = (local_features[2][i][1] - local_features[1][i][1]) / delta_t
        u = (local_features[1][i][1] - local_features[0][i][1]) / delta_t
        y_state[2,i] = (v - u) / delta_t
        
    return x_state, y_state

# Calculate the state of the current features given the previous time-step's state the current measurment
def kalman(prev_x, z, prev_p, R_cov):
    Q = np.zeros((3,3)) # process covariance
    R = R_cov # measurement covariance
    
    # state update matrix
    phi = np.array([ [1, delta_t, (delta_t*delta_t)/2],
                     [0, 1,        delta_t],
                     [0, 0,        1] ])

    H = np.identity(3) #relationship between measurement and state (in this case 1-to-1)

    # Estimate state at current time step
    x_minus = phi @ prev_x

    # Estimate the amount of error in the system at current time step
    P_minus = (phi @ prev_p @ phi.T) + Q

    # Estimate Kalman gain (weighting the current state and measurement)
    K = (P_minus @ H.T) / (H @ P_minus @ H.T + R)

    P_plus = (np.identity(3) - K @ H) @ P_minus

    x = x_minus + K @ (z - H @ x_minus)

    return x, P_plus

# Run face detection modified by the Kalman filter on a video sequence
def main(argv):

    cap = cv.VideoCapture('./videos/face_test.MOV')
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

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
    
    counter = 0
    start_frame = 3
    
    local_features = [0,0,0]
    features = []
    
    x_prev_p = np.zeros((3,3))
    y_prev_p = np.zeros((3,3))
    
    # Load measurement error covariance matrices
    x_cov = np.load("x_covariance.npy")
    y_cov = np.load("y_covariance.npy")

    for f in range(num_frames):
        # captures frame if stream not paused
        ret, frame = cap.read()

        # ret checks for correct frame read
        if not ret:
            print("Error: frame not read correctly")
            break


        # Get the x, y, and t derivatives
        grey_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        

        # Quickly find faces in a frame
        faces = detector.detectMultiScale(grey_img)

        # Detect landmarks on "grey_img"
        if len(faces) > 0:
            _, landmarks = landmark_detector.fit(grey_img, faces)
            
            features = []
            
            # Make a list of all the landmarks measurements
            for x, y in landmarks[0][0]:
                features.append( (x,y) )
                cv.circle(grey_img, (int(x), int(y)), 5, (255, 255, 255), 3)

            # Track the features of the most recent three frames
            update_features(local_features, features)

            if counter >= start_frame:
                x_measure, y_measure = format_state(local_features)
                
                # If first time step, nitialize the state to the measurment
                if counter == start_frame:
                    x_state = x_measure
                    y_state = y_measure
                # Else, run the kalman filter
                else:
                    x_state, x_prev_p = kalman(x_state, x_measure, x_prev_p, x_cov)
                    y_state, y_prev_p = kalman(y_state, y_measure, y_prev_p, y_cov)


                # display landmarks on "grey_img"
                for i in range(x_state.shape[1]):
                    cv.circle(grey_img, (int(x_state[0,i]), int(y_state[0,i])), 5, (255, 0, 0), 3)
                

        # key commands
        key = cv.waitKey(1)

        # quits if user presses q
        if key == ord('q'):
            break
            
        counter += 1

        
        cv.imshow("Video", grey_img)

    # end video stream
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
