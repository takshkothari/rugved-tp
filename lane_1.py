from re import X
from tokenize import blank_re
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math


def getLineCoordinates(frame, lines):
    slope, yIntercept = lines[0], lines[1]

    # get y and x coordinates

    y1 = frame.shape[0]
    y2 = int(y1 - 250)

    x1 = int((y1 - yIntercept) / slope)

    x2 = int((y2 - yIntercept) / slope)

    return np.array([x1, y1, x2, y2])

def getLines(frame, lines):
    copyImage = frame.copy()
    leftLine, rightLine = [], []
    lineFrame = np.zeros_like(frame)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # calculate slope & y intercept
        lineData = np.polyfit((x1, x2), (y1, y2), 1)
        slope, yIntercept = round(lineData[0], 1), lineData[1]
        if slope < 0:
            leftLine.append((slope, yIntercept))
        else:
            rightLine.append((slope, yIntercept))

    if leftLine:
        leftLineAverage = np.average(leftLine, axis=0)
        left = getLineCoordinates(frame, leftLineAverage)
        try:
            cv.line(lineFrame, (left[0], left[1]), (left[2], left[3]), (255, 0, 0), 2)
        except Exception as e:
            print('Error', e)
    if rightLine:
        rightLineAverage = np.average(rightLine, axis=0)
        right = getLineCoordinates(frame, rightLineAverage)
        try:
            cv.line(lineFrame, (right[0], right[1]), (right[2], right[3]), (255, 0, 0), 2)
        except Exception as e:
            print('Error:', e)

    return cv.addWeighted(copyImage, 0.8, lineFrame, 0.8, 0.0)


capture = cv.VideoCapture('opencv\Resources\Videos\lane_vgt.mp4')

while True:
    isTrue, frame = capture.read()

    blank = np.zeros(frame.shape[:2], dtype='uint8')

    polygon = np.array([[0, 400], [0, 300], [320, 100], [640,300], [640, 400]])
    #polygon = np.array([[0, 400], [0, 300], [125, 200], [320, 100], [525,200], [640,300], [640, 400]])

    cv.fillConvexPoly(blank, polygon, 1)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    masked_image = cv.bitwise_and(gray, gray, mask=blank)

    cv.imshow('masked', masked_image)

    gauss = cv.GaussianBlur(masked_image, (5,5), 0)

    #hsv = cv.cvtColor(gauss, cv.COLOR_BGR2HSV)

    threshold, thresh = cv.threshold(gauss, 177, 255, cv.THRESH_BINARY)

    #canny = cv.Canny(gauss, 127, 225, 5)

    #lines = cv.HoughLinesP(gauss, 1, np.pi / 180, 30, maxLineGap=200)

    #imageWithLines = getLines(frame, lines)
    #cv.imshow('final', imageWithLines)

    #cv.imshow('Video', frame)

    if cv.waitKey(10) & 0xFF==ord('0'):
        break

capture.release()
cv.destroyAllWindows()