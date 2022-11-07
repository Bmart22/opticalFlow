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
    return -local_frames[0] + local_frames[1]


def horn_schunk(x_sobel, y_sobel, t_sobel):
    u = np.zeros((x_sobel.shape[0], x_sobel.shape[1]))
    v = np.zeros((x_sobel.shape[0], x_sobel.shape[1]))

    # Calculate sobel magnitude, normalize, convert to uint8
    #    x_sobel = cv.GaussianBlur(x_sobel, (5, 5), 0, 0)
    #    y_sobel = cv.GaussianBlur(y_sobel, (5, 5), 0, 0)

    # Calculate magnitude
    sobel_mag = np.sqrt(x_sobel * x_sobel + y_sobel * y_sobel) / np.sqrt(2)
    sobel_mag = 255 * sobel_mag / np.max(sobel_mag)

    lamb = 0.5  # importance of the smoothness term
    error_cutoff = 0  # the error threshold at which we call the approximation "good enough"
    mag_cutoff = 10
    iter = 0
    num_iter = 5  # The maximum number of iterations the algorithm

    E = float('inf')

    while E > error_cutoff and iter < num_iter:
        # Calculate the average velocity within local areas
        # Use gaussian blurring to average
        ave_u = cv.GaussianBlur(u, (5, 5), 0, 0)
        ave_v = cv.GaussianBlur(v, (5, 5), 0, 0)

        # Calculate the Optical Flow Constraint (total velocity is divided between x, y, t dimensions)
        P = x_sobel * ave_u + y_sobel * ave_v + t_sobel
        # Calculate the smoothing term (encourages velocity field to be smooth)
        D = lamb * lamb + x_sobel * x_sobel + y_sobel * y_sobel
        # take ratio
        P_D = P / D

        # Calculate the velocity of the current frame
        u = ave_u - x_sobel * P_D
        v = ave_v - y_sobel * P_D

        # Calculate velocity field error term (how close are we to optimal solution?)
        new_P = x_sobel * u + y_sobel * v + t_sobel
        new_D = lamb * lamb * (u * u + v * v)
        E = (new_P * new_P + new_D).sum()

        iter += 1

    # Filter out weak optical flow
    u = np.where(sobel_mag > mag_cutoff, u, np.zeros(u.shape))
    v = np.where(sobel_mag > mag_cutoff, v, np.zeros(v.shape))

    return u, v


def plot(u, v, frame):
    # Normalize vectors
    '''u_v_mag = np.sqrt(u * u + v * v)
    non_zero_ind = np.nonzero(u_v_mag)
    u[non_zero_ind] = 5 * u[non_zero_ind] / u_v_mag[non_zero_ind]
    v[non_zero_ind] = 5 * v[non_zero_ind] / u_v_mag[non_zero_ind]'''

    # Select a subset of regularly spaced arrow to graph
    step = 25  # the stepsize we use when selecting arrow to graph
    graph_x = np.ix_(np.arange(0, u.shape[0], step), np.arange(0, u.shape[1], step))
    graph_y = np.ix_(np.arange(0, v.shape[0], step), np.arange(0, v.shape[1], step))

    # Extract the image colors to graph with the subset of arrows
    colors = frame[graph_x].transpose((1, 0, 2))
    colors = colors.reshape((colors.shape[0] * colors.shape[1], 3)) / 255

    # Graph the optical flow field, save to image
    fig, ax = plt.subplots(figsize=(4, 8))

    # inputs: (x location, y location, x orientation, y orientation, color)
    ax.quiver(np.arange(0, u.shape[1], step), np.arange(0, u.shape[0], step), u[graph_x], v[graph_y], color=colors)
    fig.savefig('thing.png')


# Main function
def main(argv):
    cap = cv.VideoCapture('./videos/IMG_7895.MOV')

    if not cap.isOpened():
        print("Error: unable to open camera")
        exit()

    is_next_frame, frame = cap.read()
    cv.imshow("Video", frame)

    # parameters:
    ksize = 3  # sobel kernel size
    ddepth = cv.CV_32F  # cv.CV_64F # sobel return type

    local_frames = [0, 0, 0]

    # Velocity components (u,v)
    u = np.zeros((frame.shape[0], frame.shape[1]))
    v = np.zeros((frame.shape[0], frame.shape[1]))

    # Need 2 frames to calculate optical flow
    counter = 0
    key = ''
    while is_next_frame and key != ord('q'):
        # captures frame if stream not paused
        is_next_frame, frame = cap.read()

        # ret checks for correct frame read
        if not is_next_frame:
            print("Frame not read correctly, or reached end of video stream.")
            break

        # Our sample videos are flipped (this info is in the video file metadata)
        frame = cv.flip(frame, 0)

        # Get the x, y, and t derivatives
        grey_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        x_sobel = cv.Sobel(grey_img, ddepth, 1, 0, ksize)
        y_sobel = cv.Sobel(grey_img, ddepth, 0, 1, ksize)

        # Add the latest frame for computing optical flow
        update_frames(local_frames, grey_img)
        if counter > 2:
            t_sobel = time_sobel(local_frames)

            # Calculate velocity vectors (u, v)
            u, v = horn_schunk(x_sobel, y_sobel, t_sobel)
            displacement = np.int64(np.stack((u, v), axis=-1))

            # Get pixel indices and where they should be displaced to
            xes = np.tile(np.arange(grey_img.shape[1]), (grey_img.shape[0], 1))
            yes = np.tile(np.arange(grey_img.shape[0])[:, None], (1, grey_img.shape[1]))
            nxes = xes + displacement[:, :, 0]
            nyes = yes + displacement[:, :, 1]

            # Keep everything within the image bounds
            nxes[(nxes < 0) | (nxes >= grey_img.shape[1])] = 0
            nyes[(nyes < 0) | (nyes >= grey_img.shape[0])] = 0

            # Place displaced pixels onto previous frame
            predicted = np.copy(grey_img)
            predicted[nyes, nxes] = grey_img

            # Save a plot of the first optical flow field
            if counter == 3:
                plot(u, v, frame)

        # Quit if user presses q
        key = cv.waitKey(1)

        # Show 3 windows: the original video stream, the corresponding predicted frames, and the difference
        cv.imshow("Video", grey_img)
        if counter > 2:
            cv.imshow("Predicted", predicted)
            cv.imshow("Difference", predicted - grey_img)

        counter += 1

    # End video stream
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
