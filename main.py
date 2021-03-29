import cv2
from motpy import MultiObjectTracker
from motpy.testing_viz import draw_detection, draw_track
from Azure_Face_Api import AzureFaceIdentify
import datetime
import time
from Utility_Face_Identification import Utility
from detection import Detectors
import logging
import argparse
from FacesDetector import FaceDetector


logger = logging.getLogger("FaceIdentification")

logfile_name = "logs\\"+datetime.datetime.now().strftime('Face_%H_%d_%m_%Y.log')
logging.basicConfig(filename=logfile_name, filemode="w", level= logging.DEBUG, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger.debug("Created log file.q")

eyes_cascade_path = "data\haarcascades\haarcascade_eye_tree_eyeglasses.xml"
eyes_cascade_path1 = "data\haarcascades\haarcascade_eye.xml"

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",  help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=700, help="minimum area size")
ap.add_argument("-f", "--cascade-face", default="data\haarcascades\haarcascade_frontalface_alt2.xml", help= "path to where the face cascade resides")
ap.add_argument("-e", "--cascade-eyes", default=eyes_cascade_path, help="path to where the eye cascade resides")
ap.add_argument("-o", "--output", default="data\Hari", help= "path to output directory")
args = vars(ap.parse_args())


try:
    def run():
        # prepare multi object tracker
        model_spec = {'order_pos': 1, 'dim_pos': 2,
                      'order_size': 0, 'dim_size': 2,
                      'q_var_pos': 5000., 'r_var_pos': 0.1}

        dt = 1 / 15.0  # assume 15 fps
        tracker = MultiObjectTracker(dt=dt, model_spec=model_spec)
        detectors = Detectors(args.get("cascade_face"), args.get("cascade_eyes"), False)
        face_detector = FaceDetector()
        faces_dict = {}
        save_img = Utility()
        logger.debug(" Initialization of classes completed.")
        logger.debug("Initializing Azure Face Identification API.")
        face_identifier = AzureFaceIdentify()

        # open camera
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)

            if not detectors.motion_detector(frame):
                time.sleep(0.5)
                logger.info("No change in frames. Waiting for 1 second before checking movement again.")
                time.sleep(1)
                continue

            logger.info("Movement detected in frame.")
            # run face detector on current frame
            detections = face_detector.process_image(frame)
            logger.info(f"{len(detections)} Faces detected in frame. ")
            tracker.step(detections)
            tracks = tracker.active_tracks(min_steps_alive=3)

            all_track_ary =[]
            if len(tracks) > 0:
                identify_faces = False
                for track in tracks:
                    all_track_ary.append(track.id)
                    if track.id in faces_dict.keys():
                        logger.info("Already detected face shown in the frame.")
                    else:
                        faces_dict[track.id] = "person data here."
                        identify_faces = True
                        logger.info("New Person entered in front of camera.")
                if identify_faces:
                    persons_identified = face_identifier.identify_persons(frame)
                    save_img.saveFrametoLocal(frame)
                    logger.info("Saving the newly entered face image for the confirmation.")
            remove_faces = []
            if len(faces_dict) > 0:
                for key in faces_dict.keys():
                    if key not in all_track_ary:
                        remove_faces.append(key)
                        logger.info("Entered Face moved out of visibility of the camera.")

            for key in remove_faces:
                del faces_dict[key]
                logger.debug("Removed face id from tracking no longer existing in front of camera.")


            # preview the boxes on frame
            for det in detections:
                draw_detection(frame, det)

            for track in tracks:
                draw_track(frame, track)

            cv2.imshow('frame', frame)
            # stop demo by pressing 'q'
            if cv2.waitKey(int(1000 * dt)) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

except Exception as e:
    logger.error("Exception raised: ", exc_info=True)

if __name__ == "__main__":
    run()
