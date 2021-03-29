from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person
import os
import time
import datetime
import logging
import sys
from Utility_Face_Identification import Utility

logger = logging.getLogger(__name__)
user_info_logger = Utility.setup_logger("Person Identifier", "PeopleEntered.log")


class AzureFaceIdentify:

    def __init__(self, KEY, ENDPOINT, PERSON_GROUP_ID="73f93a35-1b6a-4488-8272-a699ffca9b19"):
        self.face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
        self.PERSON_GROUP_ID = PERSON_GROUP_ID
        try:
            logger.info("Azure Face Api initiated.")
        # self.face_client.person_group.list()

        except Exception as e:
            logger.error("Azure call failed: ", exc_info=True)

    def identify_persons(self, image):
        logger.info("started detecting person.")
        # Detect faces
        detected_persons = {}
        face_ids = []
        # We use detection model 3 to get better performance.
        faces = self.face_client.face.detect_with_stream(image, detection_model='detection_03')
        if len(faces) > 0:
            for face in faces:
                face_ids.append(face.face_id)
        else:
            logger.warning("No Faces detected in the Frame.")
            return False

        results = self.face_client.face.identify(face_ids, self.PERSON_GROUP_ID)
        logger.info("Identifying faces in images")
        if not results:
            logger.warning("Un Identified face entered. Please register before entering. ")

        for person in results:
            if len(person.candidates) > 0:
                cur_person_id = person.candidates[0].person_id
                person_info = self.face_client.person_group_person.get(self.PERSON_GROUP_ID, cur_person_id)
                detected_persons[person_info.person_id] = {"name": person_info.name, "data": person_info.user_data}
                user_info_logger.info(f'ENTRY of {person_info.name} Identified with a confidence of {person.candidates[0].confidence}')

            else:
                logger.warning("Un Identified face entered. Please register before entering. ")

        return detected_persons

    def addNewPerson(self, person_name, person_images, person_info_dict=None):
        logger.debug("Adding new member to the group started.")
        new_person = self.face_client.person_group_person.create(self.PERSON_GROUP_ID, name=person_name,user_data=person_info_dict)
        for image in person_images:
            w = open(image, 'r+b')
            self.face_client.person_group_person.add_face_from_stream(self.PERSON_GROUP_ID, new_person.person_id, w)
        self.trainPersonGroup()

    def trainPersonGroup(self):
        logger.debug('Training the person group...')
        self.face_client.person_group.train(self.PERSON_GROUP_ID)
        while True:
            training_status = self.face_client.person_group.get_training_status(self.PERSON_GROUP_ID)
            logger.info(f'Training status for person group {self.PERSON_GROUP_ID} status is : {training_status.status}' )
            if training_status.status is TrainingStatusType.succeeded:
                break
            elif training_status.status is TrainingStatusType.failed:
                logger.error("Training the person group has failed.")
            time.sleep(5)

    def delete_person_group(self):
        try:
            self.face_client.person_group.delete(person_group_id=self.PERSON_GROUP_ID)
        except Exception as e:
            logger.error(f"Failed to delete person group please check group id {self.PERSON_GROUP_ID}",exc_info=True)
        logger.info("Deleted person group.")
