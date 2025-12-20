from bokeh.colors.groups import brown
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import torch.nn.functional as F
from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image

from triplet.triplet_model import model
from triplet.calculate_dist import orig_vector, calculate_dist


class Tracker:
    def __init__(self, path):
        self.tracker = DeepSort(max_iou_distance = 0.7, max_age = 1)
        self.path = path
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.yolo = self.load_model()
        self.transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.const = 20
        self.threshold = 75

    def load_model(self):
        model = YOLO("yolov12n-face.pt")
        model.fuse()
        model.to(self.device)
        return model

    def results(self, frame):
        results = self.yolo.predict(frame, max_det=5)[0]
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
            if not track.is_confirmed():
                continue

            bboxes = track.to_ltrb()
            idx = track.track_id
            class_id = track.get_det_class()

            results.append((bboxes, idx, class_id))

        return results

    def transform_img(self, img):
        img = Image.fromarray(img)
        img = self.transformer(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        return img

    def draw_bbox(self, bbox, frame, idx, dist, color):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"#{idx}:{dist}%"
        cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.results(frame)
            res_array = self.get_results(results, frame)

            dist = 0

            for bbox, idx, class_id in res_array:

                x1, y1, x2, y2 = map(int, bbox)

                x1 = x1 - self.const
                y1 = y1 - self.const
                x2 = x2 + self.const
                y2 = y2 + self.const

                face = frame[y1:y2, x1:x2]
                img = self.transform_img(face)


                new_vector = model.predict(img)

                new_vector = F.normalize(new_vector, dim=1)

                dist = calculate_dist(orig_vector, new_vector)
                percent_dist = (1 - round(dist.item(), 2)) * 100

                if percent_dist > self.threshold:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                self.draw_bbox(bbox, frame, idx, percent_dist, color)

            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

path = "./test_video.mp4"
tacker = Tracker(path)
tacker()