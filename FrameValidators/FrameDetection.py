from imutils.video import VideoStream
import cv2
import imutils


class FrameDetection:

    def __init__(self, display_frame=False, save_video=False):

        # initialize the first frame in the video stream
        self.original_frame = None
        self.first_frame = None
        self.current_frame = None
        self.motion_text = "Idle"
        self.frame_counter = 0
        self.is_frame_has_motion = False
        self.is_show_frame = display_frame
        self.frame_counter = 0

        if save_video:
            self.out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('F', 'M', 'P', '4'), 20.0, (500, 500))

    def handle_detection(self, frame):
        self.original_frame = frame.copy()

        self.check_frame_movement(frame)

    def check_frame_movement(self, frame):

        frame = imutils.resize(frame, width=500, height=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.first_frame is None:
            self.first_frame = gray
            return

            # compute the absolute difference between the current frame and
            # first frame
            frame_delta = cv2.absdiff(first_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

            # replace previous frame with current for future difference check.
            self.first_frame = gray

            thresh = cv2.dilate(thresh, None, iterations=2)
            contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)

            if first_contour is None:
                first_contour = contours

            if len(first_contour) == len(contours):
                frame_counter = frame_counter + 1
            else:
                first_contour = contours
                frame_counter = 0

            if frame_counter < 50:
                # loop over the contours
                for c in contours:
                    # if the contour is too small, ignore it
                    if cv2.contourArea(c) < self.frame_size:
                        continue
                    self.is_frame_has_motion = True
                    # compute the bounding box for the contour, draw it on the frame,
                    # and update the text
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    self.motion_text = "Motion Detected"

            else:
                frame_counter = 0
                self.motion_text = "Idle"
                first_frame = None
                first_contour = None

            if self.is_show_frame:
                # draw the text and timestamp on the frame
                cv2.putText(frame, "Motion Monitoring: {}".format(self.motion_text), (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

                # write frames into video
                out.write(frame)

                # show the frame and record if the user presses a key
                cv2.imshow("Motion Detection", frame)


