"""
Sources:
Face landmark detection in Python:
https://medium.com/analytics-vidhya/facial-landmarks-and-face-detection-in-python-with-opencv-73979391f30e
"""

import sys
import cv2 as cv
import numpy as np

#delta_t = 1/60
delta_t = 1


# Add newest frame to list that contains the three most recent frames
def update_frames(local_frames, new_frame):
    local_frames[0] = local_frames[1]
    local_frames[1] = local_frames[2]
    local_frames[2] = new_frame
    return
    
def update_features(local_features, new_feature):
    local_features[0] = local_features[1]
    local_features[1] = local_features[2]
    local_features[2] = new_feature
    return
    
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
        v = local_features[2][i][0] - local_features[1][i][0] / delta_t
        u = local_features[1][i][0] - local_features[0][i][0] / delta_t
        x_state[2,i] = (v - u) / delta_t
        
        v = local_features[2][i][1] - local_features[1][i][1] / delta_t
        u = local_features[1][i][1] - local_features[0][i][1] / delta_t
        y_state[2,i] = (v - u) / delta_t
        
    return x_state, y_state


def main(argv):

    cap = cv.VideoCapture('./videos/face_calibration.MOV')
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
    
    local_features = [0,0,0]
    features = []
    
    x_prev_p = np.zeros((3,3))
    y_prev_p = np.zeros((3,3))

    for f in range(num_frames):
        # captures frame if stream not paused
        ret, frame = cap.read()

        # ret checks for correct frame read
        if not ret:
            print("Error: frame not read correctly")
            break

#        frame = cv.flip(frame, 0)

        update_frames(local_frames, frame)

        # Get the x, y, and t derivatives
        grey_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        grad_x = cv.Sobel(grey_img, ddepth, 1, 0, ksize)
        grad_y = cv.Sobel(grey_img, ddepth, 0, 1, ksize)
        
        num_features = 68
        x_cov_readings = np.zeros((3, num_frames-2, num_features))
        y_cov_readings = np.zeros((3, num_frames-2, num_features))

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

            if counter >= 2:
                x_measure, y_measure = format_state(local_features)
                
#                for i in range(len(x_measure)):
                x_cov_readings[:,counter-2,:] = x_measure
                y_cov_readings[:,counter-2,:] = y_measure
                
#                # If first time step, nitialize the state to the measurment
#                if counter == 3:
#                    x_state = x_measure
#                    y_state = y_measure
#                # Else, run the kalman filter
#                else:
#                    x_state, x_prev_p = kalman(x_state, x_measure, x_prev_p)
#                    y_state, y_prev_p = kalman(y_state, y_measure, y_prev_p)
#
#
#                # display landmarks on "grey_img" with white colour in BGR and thickness 1
#                for i in range(x_state.shape[1]):
#                    cv.circle(grey_img, (int(x_state[0,i]), int(y_state[0,i])), 5, (255, 255, 255), 3)

        # key commands
        key = cv.waitKey(1)

        # quits if user presses q
        if key == ord('q'):
            break
            
        counter += 1

        
        cv.imshow("Video", grey_img)
        
    # Calculate covariance
    x_cov = np.zeros((3,3,x_cov_readings.shape[-1]))
    y_cov = np.zeros((3,3,y_cov_readings.shape[-1]))
    for i in range(x_measure.shape[-1]):
        x_cov[:,:,i] = np.cov(x_cov_readings[:,:,i])
        y_cov[:,:,i] = np.cov(y_cov_readings[:,:,i])
        
    x_cov = x_cov.mean(-1)
    y_cov = y_cov.mean(-1)
    
    
    np.save( "x_covariance.npy", x_cov )
    np.save( "y_covariance.npy", y_cov )

    # end video stream
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
