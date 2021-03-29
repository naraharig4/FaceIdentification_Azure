import cv2
import imutils


class Detectors:

    def __init__(self, cascade_face_path="data\haarcascades\haarcascade_frontalface_alt2.xml", cascade_eye_path="data\haarcascades\haarcascade_eye.xml", show_frames=False):
        self.avg = None
        self.first_frame = None
        self.face_detector = cv2.CascadeClassifier(cascade_face_path)
        self.eyes_detector = cv2.CascadeClassifier(cascade_eye_path)
        self.show_frame = show_frames

    def motion_detector(self, img):
        occupied = False
        # resize the frame, convert it to grayscale, and blur it
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        if self.first_frame is None:
            self.first_frame = gray
            return

        if self.avg is None:
            print("[INFO] starting background model...")
            self.avg = gray.copy().astype("float")

        # accumulate the weighted average between the current frame and
        # previous frames, then compute the difference between the current
        # frame and running average
        cv2.accumulateWeighted(gray, self.avg, 0.5)
        # frameDelta = cv2.absdiff(self.first_frame, gray)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg))
        # threshold the delta image, dilate the thresholded image to fill
        # in holes, then find contours on thresholded image
        thresh = cv2.threshold(frameDelta, 5, 255,
                               cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 5000:
                pass
            occupied = True

        return occupied

    def motion_detection_3frames(self,img):
        print("to be implemented.")

    def hash_face_check(self, frame1):
        rects = self.face_detector.detectMultiScale(
            cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30))

        # loop over the face detections and draw them on the frame
        for (x, y, w, h) in rects:
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the output frame
        if self.show_frame:
            cv2.imshow("HaarCascade Face Display", frame1)
        if len(rects) > 0:
            return True
        else:
            return False

    def hash_face_eyes_check(self, frame2):
        eyes = self.eyes_detector.detectMultiScale(
            cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30))
        print("detected eyes:", len(eyes))

        # loop over the face detections and draw them on the frame
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # show the output frame
        if self.show_frame:
            cv2.imshow("HaarCascade Face Display", frame2)

        if len(eyes) > 0:
            return True
        else:
            return False

    def haar_face_eyes_finder(self, frame):

        face_count = 0
        eyes_count = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)

        face_count = len(faces)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_region = gray[y:y + h, x:x + w]
            eyes = self.eyes_detector.detectMultiScale(face_region)
            # loop over the face detections and draw them on the frame
            for (x2, y2, w2, h2) in eyes:
                eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
                radius = int(round((w2 + h2) * 0.25))
                frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 1)
            eyes_count = len(eyes)
        if self.show_frame:
            cv2.imshow("HaarCascade Face Display", frame)

        return face_count, eyes_count








