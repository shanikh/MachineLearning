import cv2

img_file = 'D:\\Nick\\Documents\\car_high_1.jpg'

video = cv2.VideoCapture('D:\\YouTubeDownload\\Pedestrians Compilation.mp4')

car_tracker_file = 'D:\\Nick\\Documents\\cars_detactor.xml';
pedestrian_tracker_file = 'D:\\Nick\\Documents\\haarcascade_fullbody.xml'

car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

while True:

    (read_sucessful, frame) = video.read()
    
    if read_sucessful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrian = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

    for (x,y,w,h) in pedestrian:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)

    cv2.imshow('test car video',frame)

    cv2.waitKey(1)

video.release()

