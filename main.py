import os
from sort import Sort
from ultralytics import YOLO
import numpy as np
import cv2
import torch
from collections import deque, defaultdict
from model import model
import threading
from names import names

class Tracker:
    def __init__(self, path, device, yolo, save_dir="clips", clip_len=32):
        self.path = path
        self.yolo = yolo
        self.device = device

        self.model = self.load_model()
        self.names = self.model.names
        self.tracker = Sort(max_age=60, min_hits=5, iou_threshold=0.4)

        self.clip_len = clip_len
        self.save_dir = save_dir

        self.buffers = defaultdict(lambda: deque(maxlen=32))

        self.actions = {}
        self.action_names = names

        self.const = 20

    def load_model(self):
        model = YOLO(self.yolo)
        model.fuse()
        model.to(self.device)
        return model

    def result(self, frame):
        results = self.model.predict(source=frame,  conf=0.3, classes=[0], max_det=2)
        return results

    def get_results(self, results):
        res_array = []
        for result in results:

            if len(results) != 0:

                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                for bbox, score, class_id in zip(boxes, scores, classes):
                    arr = [bbox[0], bbox[1], bbox[2], bbox[3], score, class_id]
                    res_array.append(arr)

            return np.array(res_array)

    def procrss_clip(self, idx, clip):
        try:
            # print(f"[Thread] Start processing track {idx}")

            clip = torch.tensor(clip, dtype=torch.float32).unsqueeze(0)
            pred = model.predict(clip)

            prediction = self.action_names[int(pred)]
            self.actions[idx] = prediction
            # print(f"[Thread] Finished {idx}, prediction: {prediction}")

        except Exception as e:
            import traceback
            # print(f"[Thread ERROR] Track {idx}: {e}")
            traceback.print_exc()

    def draw(self, bboxes, idc, classes, frame):
        for bbox, idx, cls in zip(bboxes, idc, classes):
            x1, y1, x2, y2 = map(int, bbox)
            action = "Analyzing..." if idx not in self.actions else self.actions[idx]
            text = f"{idx}:{self.names[int(cls)]}:{action}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        return frame


    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            h, w, _ = frame.shape

            if not ret:
                break

            results = self.result(frame)
            res_array = self.get_results(results)

            if len(res_array) == 0:
                res_array = np.empty((0, 5))

            res = self.tracker.update(res_array)

            bboxes = res[:, :-1]
            idc = res[:, -1].astype(int)
            classes = res_array[:, -1].astype(int)

            for bbox, idx in zip(bboxes, idc):
                x1, y1, x2, y2 = map(int, bbox)

                x1 = min(0, x1 - self.const)
                y1 = min(0, y1 - self.const)

                x2 = max(w, x2 + self.const)
                y2 = max(h, y2 + self.const)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop = cv2.resize(crop, (128, 128))
                self.buffers[idx].append(crop)

                if len(self.buffers[idx]) == self.clip_len:
                    frames = list(self.buffers[idx])
                    clip = np.stack(frames, axis=0)
                    clip = np.transpose(clip, (3, 0, 1, 2))
                    # clip = torch.tensor(clip, dtype=torch.float32).unsqueeze(0)

                    threading.Thread(target=self.procrss_clip, args=(idx, clip)).start()
                    print(self.actions)


                    # pred = model.predict(clip)
                    # print(pred)

                    # self.save_clip(track_id=idx, frames=frames)
                    self.buffers[idx].clear()


            upd_frame = self.draw(bboxes, idc, classes, frame)

            cv2.imshow('Video', upd_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# path = 0
path = "videos/walking_with_dogs.mp4"
device = "mps" if torch.backends.mps.is_available() else "cpu"
yolo = "yolo11n.pt"
tracker = Tracker(path, device, yolo)

tracker()




