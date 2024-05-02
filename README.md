CS 7180

Brendan Martin, Phil Butler

Operating System: MacOS Monterey, Version 12.3

Python version: 3.9.12

Packages: Numpy, Matplotlib, OpenCV



Instructions for running programs:

To test the Kalman filter algorithm you must first run

python3 kalman_calibration.py

This generates a set of matrices which are used by kalman_test.py.


The following two functions display the optical flow and Kalman filter algorithms, respectively. In the terminal, run either:

	python3 opticalFLow.py

	python3 kalman_test.py


More information on the programs' functionality:

kalman_calibration.py: This file takes in a video which shows a stationary face; the video is used to calculate measurement error covariance matrices for the x and y axis. These matrices are saved to .npy files, which will be used by the kalman_test.py file.

opticalFLow.py: Once 3 frames have been captured, the optical flow field is computed as described above, and UI windows are opened to simultaneously display 3 things: the original video stream, the corresponding predicted frames, and their difference. One of the optical flow fields is plotted and saved into an image.

python3 kalman_test.py: This file takes in a video of a face in motion. The Facemark API and the Kalman filter are applied to this video. The video is displayed for the user, and the baseline Facemark features are drawn in white while the Kalman-modified features are drawn in blue.

