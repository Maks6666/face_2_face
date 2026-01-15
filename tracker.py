from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2 
from ultralytics import YOLO
import torch 
from torchvision import transforms
from models import age_model, gender_model, race_model, emotion_model
from PIL import Image
from connect import Table, session



class Tracker:
    def __init__(self, path: str, device: str, yolo_link: str, db: bool):
        self.path = path
        self.device = device
        self.yolo_link = yolo_link

        self.yolo = self.load_model()
        self.tracker = DeepSort(max_iou_distance = 0.7, max_age = 10)
        self.transfromer = transforms.Compose([transforms.Resize((224, 224)),
                                               transforms.ToTensor()])
        
        self.detections = {}

        self.ages = ['adults', 'childs', 'olds', 'seniors', 'teens']
        self.genders = ['female', 'male']
        self.emotions = ['anger', 'fear', 'happy', 'neutral']
        self.races = ['black', 'asian', 'white']


        age_model.to(self.device)
        gender_model.to(self.device)
        emotion_model.to(self.device)
        race_model.to(self.device)

        self.const = 30
        self.db = db
        self.prev_idx = set()
        
    
    def load_model(self):
        yolo = YOLO(self.yolo_link)
        yolo.fuse()
        yolo.to(self.device)
        return yolo
    
    def results(self, frame):
        results = self.yolo.predict(frame, max_det=5, verbose=False)[0]
        return results
    
    def get_results(self, results, frame):
        res_array = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            res_array.append((bbox, float(score), int(class_id)))
        
        tracks = self.tracker.update_tracks(raw_detections=res_array, frame=frame)
        results = []

        for track in tracks:
            bbox = track.to_ltrb()
            idx = track.track_id
            class_id = track.get_det_class()


            results.append((bbox, idx, class_id))
        
        return results
    
    def transfrom(self, bbox, frame):
        x1, y1, x2, y2 = map(int, bbox)

        h, w = frame.shape[:2]

        x1 = max(0, x1 - self.const)
        y1 = max(0, y1 - self.const)
        x2 = min(w, x2 + self.const)
        y2 = min(h, y2 + self.const)

        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        crop = frame[y1:y2, x1:x2]
        img = Image.fromarray(crop)
        img = self.transfromer(img).unsqueeze(0)
        img = img.to(self.device)
        return img
    
    def draw(self, bbox, idx, frame):
        x1, y1, x2, y2 = map(int, bbox)

        age, gender, emotion, race = "Analysing...", "Analysing...", "Analysing...", "Analysing..." 

        if idx in self.detections:
            data = self.detections[idx]

            if len(data) == 4:
                age = self.ages[int(data[0])]
                gender = self.genders[int(data[1])]
                emotion = self.emotions[int(data[2])]
                race = self.races[int(data[3])]
            

        text_color = (0, 0, 255)
        frame_color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), frame_color, 2)
        id = f"#{idx}"
        cv2.putText(frame, id, (x1, y1-130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)

        cv2.putText(frame, age, (x1+80, y1-130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
        cv2.putText(frame, gender, (x1+80, y1-90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
        cv2.putText(frame, emotion, (x1+80, y1-50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
        cv2.putText(frame, race, (x1+80, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)


        return frame
    
    def save_to_db(self, idx) -> None:
        if idx in self.detections:
            data = self.detections[idx]
            if len(data) == 4:
                age = self.ages[int(data[0])]
                emotion = self.emotions[int(data[2])]
                gender = self.genders[int(data[1])]
                race = self.races[int(data[3])]


            existing = session.query(Table).filter_by(idx=idx).first()
            if not existing:
                row = Table(idx=idx, age=age, emotion=emotion, gender=gender, race=race)
                session.add(row)
            session.commit()

    def delete_from_db(self, idx) -> None:
        row = session.query(Table).filter_by(idx=idx).first()
        if row:
            session.delete(row)
            session.commit()
    
    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            if not ret: 
                break

            results = self.results(frame=frame)
            res_array = self.get_results(results=results, frame=frame)

            current_idx = set()

            for bbox, idx, _ in res_array:
                current_idx.add(idx)

                frame = self.draw(bbox, idx, frame)
                img = self.transfrom(bbox, frame)

                age = age_model.predict(img)
                gender = gender_model.predict(img)
                emotion = emotion_model.predict(img)
                race = race_model.predict(img)

                self.detections[idx] = [age, gender, emotion, race]

                if self.db:
                    self.save_to_db(idx)


            disappeared_idx = self.prev_idx - current_idx
            # print(disappeared_idx)
            for idx in disappeared_idx:
                self.delete_from_db(idx)
                self.detections.pop(idx, None)

            
            self.prev_idx = current_idx

            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



# path = "./video.mp4"
# device = "mps" if torch.backends.mps.is_available() else "cpu" 
# yolo_link = "./yolov12s-face.pt"
# save_to_db = True

# tracker = Tracker(path, device, yolo_link, save_to_db)
# tracker()





        
