import cv2
import math
import pandas as pd
from segmentation import segment
import SerialArduino
from ultralytics import YOLO

# Load a model
model = YOLO('best.pt')  # load a custom model

# Predict with the model
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image

# Baca video
video_top = 'Camera_B_21.mp4'
cap = cv2.VideoCapture(video_top)

video_side = 'Camera_A_21.mp4'
cap2 = cv2.VideoCapture(video_side)

count = 0
midold = []
cid = [-1]
trackob = {}
track_id = 0
t = 0
buah = 0

while cap.isOpened():
    count += 1
    ret, frame = cap.read()
    _, frame2 = cap2.read()
    frame2 = frame2[200:800, 600:1200]
    if not ret:
        break
    # Ubah ruang warna BGR ke HSVC
    result = segment(frame)
    result2 = segment(frame2)

    #cv2.line(frame, (0, 540), (1920, 540), (60, 35, 176), 10)

    midnow = []


    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(result2, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours2, hierarchy2 = cv2.findContours(gray2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)



    for cnt2 in contours2:
        # Calculate area and remove small elements
        area2 = cv2.contourArea(cnt2)
        if area2 > 15000:
            x2, y2, w2, h2 = cv2.boundingRect(cnt2)
            cv2.rectangle(frame2, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 225), 3)
            cv2.putText(frame2, str(h2), (x2, y2 + h2), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)
            t = h2

    for cnt in contours:
            # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100000 and area < 380000:
            x, y, w, h = cv2.boundingRect(cnt)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 225), 3)
            #cv2.putText(frame, str(w * h), (x, y + h), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)


            cv2.putText(frame, (x, y + h), cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 0), 10)

            midnow.append((cx, cy, w, h, t))



    if count <= 2:
        for pt in midnow:
            for pt2 in midold:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                if distance < 50:
                    trackob[track_id] = pt
                    track_id += 1

    else:
        trackob_copy = trackob.copy()
        midnow_copy = midnow.copy()

        for object_id, pt2 in trackob_copy.items():
            object_exists = False
            for pt in midnow_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # update object position
                if distance < 50:
                    trackob[object_id] = pt
                    object_exists = True
                    if pt in midnow:
                        midnow.remove(pt)
                        continue

            # remove id
            if not object_exists:
                trackob.pop(object_id)

        for pt in midnow:
            trackob[track_id] = pt
            track_id += 1

    for object_id, pt in trackob.items():
        #cv2.putText(frame, str(object_id), (pt[0], pt[1] + 7), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        if (object_id not in cid and 510 < pt[1] and pt[1] < 530):
            buah =+1
            cid.append(object_id)
            image = frame[pt[1]-320:pt[1]+320, pt[0]-320:pt[0]+320]
            data = [pt[2], pt[3], pt[4]]
            print(data)
            cv2.imwrite("image.jpg", image)
            results = model('image.jpg')


        if (pt[1] == 540 and buah % 3 == 0):
            if cid:
                cid.pop(0)


    midold = midnow.copy()

    cv2.imshow("Result", frame)
    cv2.imshow("Result2", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap2.release()
cv2.destroyAllWindows()
