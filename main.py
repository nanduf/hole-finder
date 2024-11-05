import cv2
import numpy as np
import math

filename = 'subsquare.jpg'


#read in images
img = cv2.imread(filename, cv2.IMREAD_COLOR)
bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread('holetemplate.jpg', cv2.IMREAD_COLOR)
bw_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

#resizing template image if needed
if(filename=='square.jpg'):
    dsize = (40, 40)
    bw_template = cv2.resize(bw_template,dsize)
elif(filename=='subsquare.jpg'):
    dsize = (240, 240)
    bw_template = cv2.resize(bw_template, dsize)
w, h = bw_template.shape[::-1]


res = cv2.matchTemplate(bw_img, bw_template, cv2.TM_CCOEFF_NORMED)
threshold = .5
loc = np.where(res >= threshold)
scores = [res[x, y] for x, y in zip(*loc)]

loc2 = (loc[1], loc[0])  #reverse x and y, (y, x) -> (x, y)
points = loc2 + (scores,) #add corresponding score to each coordinate

selected_points = []
selected_scores = []
circle = []
circle_scores = []
radius = h//2
for pt1 in zip(*points):
    point = (pt1[0] + w // 2, pt1[1] + h // 2)
    circle.append(point)
    circle_scores.append(pt1[2])
    for pt in circle:
        hypot = np.array(point) - np.array(pt)
        if math.sqrt(pow(hypot[0], 2) + pow(hypot[1], 2)) > radius:
            circle.pop()
            circle_scores.pop()
            max_index = circle_scores.index(max(circle_scores))
            selected_scores.append(max(circle_scores))
            selected_points.append(circle[max_index])
            circle = []
            circle_scores = []
            break

final_points = []
final_scores = []
for spt in selected_points:
    final_points.append(spt)
    final_scores.append(selected_scores[selected_points.index(spt)])
    for fpt in final_points:
        hypot = np.array(spt) - np.array(fpt)
        if math.sqrt(pow(hypot[0], 2) + pow(hypot[1], 2)) < radius: # combine
            if(final_scores[final_points.index(fpt)] < final_scores[final_points.index(spt)]):
                index = final_points.index(fpt)
                final_points.pop(index)
                final_scores.pop(index)
            elif(final_scores[final_points.index(fpt)] > final_scores[final_points.index(spt)]):
                index = final_points.index(spt)
                final_points.pop(index)
                final_scores.pop(index)

#drawing circles and '+'
for pt in final_points:
    center = (pt[0], pt[1])
    confidence = (1 - final_scores[final_points.index(pt)]) * (1 / (1-threshold)) * 100#lower number = better score/confidence
    cv2.circle(img, center, h//4, (0, 0, 255), 1)
    cv2.line(img, (center[0] - h//5, center[1]), (center[0] + h//5, center[1]), (0, 0, 0), 1)
    cv2.line(img, (center[0], center[1] - h//5), (center[0], center[1] + h//5), (0, 0, 0), 1)

#showing image
cv2.imshow("Detected Circles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
