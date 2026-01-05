import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import numpy as np
import cv2
import torch
from collections import deque, defaultdict
from model import TeacherNet, StudentNet
import threading
from names import names
from db import session, Table

class Tracker:
    def __init__(self, path, yolo, model_link, device, action_device, save):
        self.path = path
        self.yolo = yolo 
        self.device = device
        self.action_device = action_device
        self.detector = self.load_model()
        self.tracker = DeepSort(max_iou_distance = 0.4, max_age = 5, n_init=1)

        self.model_link = model_link
        self.action_model = self.load_action_model()
        self.buffers = defaultdict(lambda: deque(maxlen=16))
        self.actions = {}
        self.const = 90

        self.save = save
        
    
    def load_model(self):
        model = YOLO(self.yolo)
        model.fuse()
        model.to(self.device)
        return model

    def results(self, frame):
        results = self.detector.predict(source=frame,  conf=0.3, classes=[0], max_det=2, verbose=False)[0]
        return results

        
    
    def get_results(self, results, frame):
        res_array = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            res_array.append((bbox, float(score), int(class_id)))

        # print("RAW DETS:", res_array)
        tracks = self.tracker.update_tracks(raw_detections = res_array, frame=frame)
        outputs = []

        for track in tracks:
            if not track.is_confirmed():
                continue
        
            bboxes = track.to_ltrb()
            idx = track.track_id
            class_id = track.get_det_class()

            outputs.append((bboxes, idx, class_id))
        
        return outputs
    
    def preprocess_clip(self, idx, clip):
        clip = torch.tensor(clip, dtype=torch.float32).unsqueeze(0) / 255.0
        clip = clip.to(self.action_device)
        pred = self.action_model.predict(clip)
        action = names[pred]
        self.actions[idx] = action

    def resize_frame(self, bbox, frame):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]

        x1 = max(0, x1 - self.const)
        y1 = max(0, y1 - self.const)
        x2 = min(w, x2 + self.const)
        y2 = min(h, y2 + self.const)

        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        return x1, y1, x2, y2
        
    def draw(self, bbox, idx, frame):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        action = "Analysing..." if idx not in self.actions else self.actions[idx]
        text = f"{idx}:{action}"
        cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        return frame
    
    def save_to_db(self) -> None:
        if len(self.actions) > 0:
            for key, value in self.actions.items():
                idx = key
                action = value

                existing = session.query(Table).filter_by(idx=idx).first()
                if not existing:
                    row = Table(idx=idx, action=action)
                    session.add(row)
                session.commit()

    
    def load_action_model(self):
        try:
            # model = TeacherNet()
            model = StudentNet()
            model.load_state_dict(torch.load(self.model_link, map_location=self.action_device))
            model.to(self.action_device)  
            return model
        except Exception as e:
            raise e
    

    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            if not ret: 
                break

            results = self.results(frame=frame)
            res_array = self.get_results(results=results, frame=frame)

            for bbox, idx, _ in res_array:
                frame = self.draw(bbox, idx, frame)
    
                x1, y1, x2, y2 = self.resize_frame(bbox, frame)

                clip = frame[y1:y2, x1:x2]

                if clip.size == 0:
                    continue
                
                clip = cv2.resize(clip, (128, 128))
                self.buffers[idx].append(clip)
                if len(self.buffers[idx]) == 16:
                    clip = list(self.buffers[idx])
                    clip = np.stack(clip, axis=0)
                    clip = np.transpose(clip, (0, 3, 1, 2))
                    # print(clip.shape)

                    threading.Thread(target=self.preprocess_clip, args=(idx, clip)).start()
                    # print(self.actions)
                    self.buffers[idx].clear()
            


            if self.save:
                self.save_to_db()
        

            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# path = "./test.mp4"
# yolo = "yolo12l.pt"
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# tracker = Tracker(path, yolo, model, device)
# tracker()

            




