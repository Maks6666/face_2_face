import cv2
import torch
from ultralytics import YOLO
import numpy as np
from sort import Sort
from time import time
from models import age_model, gender_model, race_model, emotion_model
from connect import get_data, get_idx, Session, Table, IndexTable, detected_filter
from tools import return_face

class ObjectDetection:
    def __init__(self, capture_index):
        self.session = Session()
        self.capture_index = capture_index
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.face_model = self.load_face_model()
        self.class_names = ["person"]


    def load_face_model(self):
        model = YOLO("weights/yolov10n-face.pt")
        model.fuse()
        return model

    def predict(self, frame, model):
        results = model.predict(frame, verbose=True, conf=0.4)
        return results

    def get_results(self, results):
        detections_list = []

        for result in results[0]:

            class_id = result.boxes.cls.cpu().numpy()

            if class_id == 0:
                bbox = result.boxes.xyxy.cpu().numpy()
                confidence = result.boxes.conf.cpu().numpy()
                class_id = result.boxes.cls.cpu().numpy()
                merged_detection = [bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3], confidence[0], class_id[0]]
                detections_list.append(merged_detection)
        return np.array(detections_list)


    def face_results(self, results):
        face_list = []
        for result in results[0]:
            bbox = result.boxes.xyxy.cpu().numpy()
            confidence = result.boxes.conf.cpu().numpy()
            class_id = result.boxes.cls.cpu().numpy()
            coordinates = [bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3], confidence[0], class_id[0]]
            face_list.append(coordinates)
        return np.array(face_list)




    def draw_bounding_boxes_with_id(self, img, face_bboxes, face_ids, class_id):
        for i, (face_box, face_id, cls) in enumerate(zip(face_bboxes, face_ids, class_id)):
            t_face_id = int(face_id)
            # t_obj - is a WHOLE ROW
            t_obj = self.session.query(Table).filter_by(object_id=t_face_id).first()

            t_label = "Analyzing..."

            id_label = ""
            age_lable = ""
            gen_label = ""
            race_label = ""
            emotion_label = ""

            # TO TAKE ONLY ONE VALUE FROM ROW USE t_obj.object_id
            if t_obj:
                if t_obj.object_id == face_id:
                    t_label = ""
                    id_label = f"#: {face_id}"
                    age_lable = f"Age: {t_obj.age_category}"
                    gen_label =  f"Gen: {t_obj.gender}"
                    race_label = f"Race: {t_obj.race}"
                    emotion_label = f"Emotion: {t_obj.face}"


            # print(f"Label for ID {face_id}: {label}")

            cv2.rectangle(img, (int(face_box[0]) - 50, int(face_box[1]) - 50), (int(face_box[2]) + 50, int(face_box[3]) + 50), (0, 0, 255), 2)
            cv2.putText(img, t_label, (int(face_box[0]), int(face_box[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.putText(img, id_label, (int(face_box[0]), int(face_box[1]) - 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
            cv2.putText(img, age_lable, (int(face_box[0]), int(face_box[1]) - 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
            cv2.putText(img, gen_label, (int(face_box[0]), int(face_box[1]) - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
            cv2.putText(img, race_label, (int(face_box[0]), int(face_box[1]) - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
            cv2.putText(img, emotion_label, (int(face_box[0]), int(face_box[1]) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

        return img


    # Custom models deployment:
#   ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def get_age(self, data):

        if data is None or data.size == 0 or len(data) == 0:
            return None

        age_list = ['adults (26 - 45)', 'children (7 - 13)', 'middles (46 - 60)', 'seniors (61-90)', 'teens (14 - 25)',
                    'toddlers (1 - 6)']

        face = return_face(data, self.device)

        res = age_model(face)

        fin_res = age_list[int(res)]

        return fin_res, int(res)


    def get_gender(self, data):

        if data is None or data.size == 0 or len(data) == 0:
            return None

        gender_list = ["female", "male"]

        face = return_face(data, self.device)

        res = gender_model(face)

        fin_res = gender_list[int(res)]

        return fin_res, int(res)

    def get_race(self, data):

        if data is None or data.size == 0 or len(data) == 0:
            return None

        race_list = ["negroid", "mongoloid", "europeoid"]

        face = return_face(data, self.device)

        res = race_model(face)

        fin_res = race_list[int(res)]

        return fin_res, int(res)

    def get_emotion(self, data):
        if data is None or data.size == 0 or len(data) == 0:
            return None

        emotions = ['anger', 'fear', 'happy', 'neutral']

        face = return_face(data, self.device)

        res = emotion_model(face)

        fin_res = emotions[int(res)]

        return fin_res, int(res)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        face_sort = Sort(max_age=100, min_hits=8, iou_threshold=0.30)


        while True:
            detected_idx = []
            objects_to_delete = []

            start_time = time()
            ret, frame = cap.read()
            assert ret

            face_results_list = self.predict(frame, self.face_model)
            face_results = self.face_results(face_results_list)

            if len(face_results) == 0:
                face_results = np.empty((0, 5))

            f_res = face_sort.update(face_results)

            f_boxes_track = f_res[:, :-1]
            f_boxes_ids = f_res[:, -1].astype(int)
            f_class_id = face_results[:, -1].astype(int)

            for bbox, ids, cls in zip(f_boxes_track, f_boxes_ids, f_class_id):
                detected_idx.append(ids)
                x1, y1, x2, y2 = map(int, bbox)

                new_x1 = x1  - 20
                new_y1 = y1 - 20

                new_x2 = x2 + 20
                new_y2 = y2 + 20

                cropped_face = frame[new_y1:new_y2, new_x1:new_x2]

                # age, age_idx = self.get_age(cropped_face)
                # gender, gender_idx  = self.get_gender(cropped_face)
                # race, race_idx = self.get_race(cropped_face)
                # emotion, emotion_idx = self.get_emotion(cropped_face)

                age_res = self.get_age(cropped_face)
                gender_res = self.get_gender(cropped_face)
                race_res = self.get_race(cropped_face)
                emotion_res = self.get_emotion(cropped_face)


                if age_res is not None and gender_res is not None and race_res is not None and  emotion_res is not None:
                    age, age_idx = age_res
                    gender, gender_idx = gender_res
                    race, race_idx = race_res
                    emotion, emotion_idx = emotion_res


                #
                # if age_res is not None and gender_res is not None and race_res is not None and  emotion_res is not None:
                #     age, age_idx = age_res
                #
                # if gender_res is not None:
                #     gender, gender_idx = gender_res
                #
                # if race_res is not None:
                #     race, race_idx = race_res
                #
                # if emotion_res is not None:
                #     emotion, emotion_idx = emotion_res




                # fin_age = get_or_create_age(self.session, ids, age)

                # I copied 'status' column to TotalIndexTable table from IndexTable where all values
                # are correctly sorted
                    total_obj_list = self.session.query(IndexTable).filter_by(status="detected").all()
                    total_obj = len(total_obj_list)


                    get_data(self.session, ids, int(cls), self.class_names[int(cls)], age, gender, race, emotion)
                    get_idx(self.session, ids, int(cls), age_idx, gender_idx, race_idx, emotion_idx, total_obj)
                    # get_total_info(self.session, total_obj)


            detected_filter(self.session, objects_to_delete, detected_idx)

            frame = self.draw_bounding_boxes_with_id(frame, f_boxes_track, f_boxes_ids, f_class_id)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if cv2.waitKey(1) & 0xFF == ord("d"):
                self.session.query(Table).delete()
                self.session.query(IndexTable).delete()
                self.session.commit()
                self.session.close()
                break

        cap.release()
        cv2.destroyAllWindows()

"ages_video.mp4"
"videos/ages_video.mp4"
"videos/test_video_3.mp4"
"videos/test_video_4.mp4"
detector = ObjectDetection(capture_index="videos/ages_video.mp4")
detector()

