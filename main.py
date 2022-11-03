import sys
import cv2 as cv

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
    ave_u = cv2.GaussianBlur(prev_u, (5, 5), 0, 0)
    ave_v = cv2.GaussianBlur(prev_v, (5, 5), 0, 0)
    
    lamb = 0.5 # importance of the smoothness term
    error_cutoff =
    iter = 0
    
    while (E > error_cutoff && iter < num_iter) :
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
        
        iter++
        
    # Filter out weak optical flow
    
    return



def main(argv):
    if len(argv) == 1:
        capdev = cv.VideoCapture(0)
    elif len(argv) == 2:
        if argv[1] == 'windows':
            capdev = cv.VideoCapture(0, cv.CAP_DSHOW)
        elif argv[1] == 'webcam':
            capdev = cv.VideoCapture(1)
    else:
        print("Invalid number of arguments. Expected 0 or 1. See readme.md for details.")

    if not capdev.isOpened():
        print("Error: unable to open camera")
        exit()

    ret, frame = capdev.read()
    cv.imshow("Video", frame)
    cv.setMouseCallback('Video', click_event, param=[frame])
    
    
    # parameters:
    ksize = 3 # sobel kernel size
    ddepth = -1 #cv2.CV_64F # sobel return type
    
    local_frames = [0,0,0]

    while True:
        # captures frame if stream not paused
        ret, frame = capdev.read()

        # ret checks for correct frame read
        if ret is not True:
            print("Error: frame not read correctly")
            exit()
            
        update_frames(local_frames, frame)
        
        
        
        # Get the x, y, and t derivatives
        grey_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
        grad_x = cv2.Sobel(grey_img, ddepth, 1, 0, ksize);
        grad_y = cv2.Sobel(grey_img, ddepth, 0, 1, ksize);
        
        
        

        # key commands
        key = cv.waitKey(1)

        # quits if user presses q
        if key == ord('q'):
            break

        cv.imshow("Video", frame)

    # end video stream
    capdev.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
