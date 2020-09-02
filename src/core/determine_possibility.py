import numpy as np
import cv2

# --------------------------------

def convertColorSpace(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def threshold(frame):
    # HSV 색공간을 이용. 채도와 밝기가 낮은 곳을 선택함. (==그림자)
    range_lower = (  0,  0,  0)
    range_upper = (255, 96, 92)
    return cv2.inRange(frame, range_lower, range_upper)

def morphology(frame_bin):
    frame_bin = cv2.morphologyEx(frame_bin, cv2.MORPH_CLOSE, (5,5))
    return cv2.morphologyEx(frame_bin, cv2.MORPH_OPEN, (5,5))

def contourAreaThreshold(frame_bin):
    MIN_CONT_AREA = 20
    # 윤곽선 면적에도 임계를 주어 훼손여부를 가림.
    colored_mask = cv2.cvtColor(frame_bin, cv2.COLOR_GRAY2BGR)
    for contour in cv2.findContours(frame_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]:
        # 임계치를 넘는다면, 붉은 색으로 처리: 훼손된 영역
        if (cv2.contourArea(contour) >= MIN_CONT_AREA):
            cv2.drawContours(colored_mask, [contour], -1, (0,0,255), -1)
        # 임계치를 넘지 못한다면, (반투명)노란색으로 처리
        else:
            cv2.drawContours(colored_mask, [contour], -1, (0,128,128), -1)
    return colored_mask

# --------------------------------

sequence = [convertColorSpace, threshold, morphology, contourAreaThreshold]