import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Add newest frame to list that contains the three most recent frames
def update_frames(local_frames, new_frame):
    local_frames[0] = local_frames[1]
    local_frames[1] = local_frames[2]
    local_frames[2] = new_frame
    return

# Calculate the time-dimension sobel filter
def time_sobel(local_frames):
#    print(local_frames[0].shape)
    
    # Init output array
    dim = local_frames[0].shape
    filter_output = np.zeros( dim )
    
    # Loop over all pixels
#    for r in range(dim[0]):
#        for c in range(dim[1]):
    filter_output = -local_frames[0] + local_frames[1]
        
    return filter_output
    
def horn_schunk(x_sobel, y_sobel, t_sobel):
    u = np.zeros((x_sobel.shape[0],x_sobel.shape[1]))
    v = np.zeros((x_sobel.shape[0],x_sobel.shape[1]))

    
    
    # Calc sobel magnitude, normalize, convert to uint8
#    x_sobel = cv.GaussianBlur(x_sobel, (5, 5), 0, 0)
#    y_sobel = cv.GaussianBlur(y_sobel, (5, 5), 0, 0)
    
    
    # Calculate magnitude
    sobel_mag = np.sqrt(x_sobel*x_sobel + y_sobel*y_sobel) / np.sqrt(2)
    sobel_mag = 255 * sobel_mag / np.max(sobel_mag)

    lamb = 0.5 # importance of the smoothness term
    error_cutoff = 0 # the error threshold at which we call the approximation "good enough"
    mag_cutoff = 10
    iter = 0
    num_iter = 5 # The maximum number of iterations the algorithm
    
    E = float('inf')

    while (E > error_cutoff and iter < num_iter) :
        # Calculate the average velocity within local areas
        # Use gaussian blurring to average
        ave_u = cv.GaussianBlur(u, (5, 5), 0, 0)
        ave_v = cv.GaussianBlur(v, (5, 5), 0, 0)
        
        # Calculate the Optical Flow Constraint (total velocity is divided between x, y, t dimensions)
        P = x_sobel*ave_u + y_sobel*ave_v + t_sobel
        # Calculate the smoothing term (encourages velocity field to be smooth)
        D = lamb*lamb + x_sobel*x_sobel + y_sobel*y_sobel
        # take ratio
        P_D = P/D

        # Calculate the velocity of the current frame
        u = ave_u - x_sobel * P_D
        v = ave_v - y_sobel * P_D

        # Calculate velocity field error term (how close are we to optimal solution?)
        new_P = x_sobel*u + y_sobel*v + t_sobel
        new_D = lamb*lamb * (u*u + v*v)
        E = (new_P*new_P + new_D).sum()

        iter += 1

    # Filter out weak optical flow
    u = np.where( sobel_mag > mag_cutoff, u, np.zeros(u.shape) )
    v = np.where( sobel_mag > mag_cutoff, v, np.zeros(v.shape) )

    return u, v
#
#def kalman(prev_x, prev_z, P_minus):
#    #x = [pos, vel, acc] # state
#    Q = np.zeros((3,3)) # process covariance
#    R = np.zeros((3,3)) # measurement covariance
#
#    phi = np.array([ [1, delta, (delta*delta)/2]
#                     [0, 1,      delta],
#                     [0, 0,      1] ])
#
#    H = np.identity(3) #relationship between measurement and state (in this case 1-to-1)
#
#    # Estimate state at current time step
#    x = phi @ prev_x
#
#    # Estimate the amount of error in the system as current time step
#    P_minus = (phi @ prev_p @ phi.T) + Q
#
#    # Estimate Kalman gain (weighting the current state and measurement)
#    K = (P_minus @ H.T) / (H @ P_minus @ H.T + R)
#
#    P_plus = (np.identity(3) - K @ H) @ P_minus
#
#    x = x_minus + K @ ( z - H @ x_minus )
#
#    return x, P_plus
    
    

def main(argv):
    cap = cv.VideoCapture('./videos/IMG_7896.MOV')

    if not cap.isOpened():
        print("Error: unable to open camera")
        exit()

    ret, frame = cap.read()
    cv.imshow("Video", frame)
#    cv.setMouseCallback('Video', click_event, param=[frame])
    
    
    # parameters:
    ksize = 3 # sobel kernel size
    ddepth = cv.CV_32F #cv.CV_64F # sobel return type
    
    local_frames = [0,0,0]
    
    # Velocty components (u,v)
    u = np.zeros((frame.shape[0],frame.shape[1]))
    v = np.zeros((frame.shape[0],frame.shape[1]))
    
    counter = 0

    while True:
        # captures frame if stream not paused
        ret, frame = cap.read()

        # ret checks for correct frame read
        if ret is not True:
            print("Error: frame not read correctly")
            exit()
            
            
        frame = cv.flip(frame, 0)
        
        # Get the x, y, and t derivatives
        grey_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        x_sobel = cv.Sobel(grey_img, ddepth, 1, 0, ksize)
        y_sobel = cv.Sobel(grey_img, ddepth, 0, 1, ksize)
        
        update_frames(local_frames, grey_img)
        
        if counter > 2:
            t_sobel = time_sobel(local_frames)
            
            # Calculate veclocity vectors (u, v)
            u,v = horn_schunk(x_sobel, y_sobel, t_sobel)
            
            # Normalize vectors
            u_v_mag = np.sqrt(u*u + v*v)
            non_zero_ind = np.nonzero(u_v_mag)
            u[non_zero_ind] = 5*u[non_zero_ind] / u_v_mag[non_zero_ind]
            v[non_zero_ind] = 5*v[non_zero_ind] / u_v_mag[non_zero_ind]
            
            # Select a subset of regularly spaced arrow to graph
            step = 25 # the stepsize we use when selecting arrow to graph
            graph_x = np.ix_( np.arange(0, u.shape[0], step), np.arange(0, u.shape[1], step) )
            graph_y = np.ix_( np.arange(0, v.shape[0], step), np.arange(0, v.shape[1], step) )

            # Extract the image colors to graph with the subset of arrows
#            print(len(graph_x))
#            print(graph_x[0].shape)
            colors = frame[graph_x].transpose((1,0,2))
            
            colors = colors.reshape((colors.shape[0]*colors.shape[1],3)) / 255
#            colors = frame[graph_x].transpose((1,0,2)) / 255
            
            # Graph the optical flow field, save to image
            fig, ax = plt.subplots( figsize=(4,8) )
            
            print(colors.shape)
            print(u[graph_x].shape)
            print(v[graph_y].shape)
            
            # inputs: (x location, y location, x orientation, y orientation, color)
            ax.quiver(np.arange(0, u.shape[1], step), np.arange(0, u.shape[0], step), u[graph_x],v[graph_y], color = colors)
            fig.savefig('thing.png')
            
        

        # key commands
        key = cv.waitKey(1)

        # quits if user presses q
        if key == ord('q'):
            break

        grey_img = cv.flip(grey_img, 0)
        cv.imshow("Video", grey_img)
        
        counter += 1

    # end video stream
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
