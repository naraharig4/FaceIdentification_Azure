from motpy.detector import BaseObjectDetector



class FaceDetector(BaseObjectDetector):
    def __init__(self,
                 weights_url: str = WEIGHTS_URL,
                 weights_path: str = WEIGHTS_PATH,
                 config_url: str = CONFIG_URL,
                 config_path: str = CONFIG_PATH,
                 conf_threshold: float = 0.5) -> None:
        super(FaceDetector, self).__init__()

        if not os.path.isfile(weights_path) or not os.path.isfile(config_path):
            logger.debug('downloading model...')
            urlretrieve(weights_url, weights_path)
            urlretrieve(config_url, config_path)

        self.net = cv2.dnn.readNetFromCaffe(config_path, weights_path)

        # specify detector hparams
        self.conf_threshold = conf_threshold

    def process_image(self, image: NpImage) -> Sequence[Detection]:
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        self.net.setInput(blob)
        detections = self.net.forward()

        # convert output from OpenCV detector to tracker expected format [xmin, ymin, xmax, ymax]
        out_detections = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                xmin = int(detections[0, 0, i, 3] * image.shape[1])
                ymin = int(detections[0, 0, i, 4] * image.shape[0])
                xmax = int(detections[0, 0, i, 5] * image.shape[1])
                ymax = int(detections[0, 0, i, 6] * image.shape[0])
                out_detections.append(Detection(box=[xmin, ymin, xmax, ymax], score=confidence))

        return out_detections

