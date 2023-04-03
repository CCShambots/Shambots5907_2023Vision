import cv2 as cv
import numpy as np
import math


def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    # [visualization1]
    angle = math.atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = math.sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * math.cos(angle)
    q[1] = p[1] - scale * hypotenuse * math.sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * math.cos(angle + math.pi / 4)
    p[1] = q[1] + 9 * math.sin(angle + math.pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

    p[0] = q[0] + 9 * math.cos(angle - math.pi / 4)
    p[1] = q[1] + 9 * math.sin(angle - math.pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
    # [visualization1]


def getOrientation(pts, img):
    # [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]): 
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    # [pca]

    # [visualization]
    # Draw the principal components
    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
        cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
        cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
        cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
        cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (255, 255, 0), 1)
    drawAxis(img, cntr, p2, (0, 0, 255), 5)

    angle = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    # [visualization]

    # Label with the rotation angle
    label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    textbox = cv.rectangle(img, (cntr[0], cntr[1] - 25), (cntr[0] + 250, cntr[1] + 10), (255, 255, 255), -1)
    cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

    return angle


def runPipeline(image, llrobot):
    imgHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # convert the hsv to ab inary image by removing pixels
    # not in the HSV min/max values
    imgThresh = cv.inRange(imgHSV, (25, 100, 0), (35, 255, 255))

    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5, 5), np.uint8)

    # Erode and dilate
    cv.erode(imgThresh, kernel, 2)
    cv.dilate(imgThresh, kernel, 2)

    contours, hierarchy = cv.findContours(imgThresh,
                                          cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    largestContour = np.array([[]])

    # initialize empty array to send data to bot
    llpython = [0]

    # if contours have been detected, draw them
    if len(contours) > 0:
        cv.drawContours(image, contours, -1, 255, 2)

        largestContour = max(contours, key=cv.contourArea)

        angle = getOrientation(largestContour, image)

        llpython = [angle]

    cv.imshow('Stream', image)

    return largestContour, image, llpython


# cap = cv.VideoCapture("cone.mkv")
#
# if not cap.isOpened():
#     exit
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         # contour, showFrame, data = runPipeline(frame, 0)
#         cv.imshow('Stream', frame)
