import argparse
import cv2
import torch 
import torch.nn.functional as F
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import transforms
from PIL import Image

from triplet.triplet_model import triplet
from triplet.calculate_dist import calculate_dist, orig_vector

def parse_args():
    parser = argparse.ArgumentParser(description="Face detection & recognition inference")
    parser.add_argument("--video", type=str, required=True, help="Path to videofile")
    parser.add_argument("--yolo_weights", type=str, default="yolov12n-face.pt", help="YOLO face weights")
    parser.add_argument("--threshold", type=float, default=75.0, help="Similarity threshold")
    parser.add_argument("--device", type=str, default="mps", help="cpu | cuda | mps")
    return parser.parse_args()

def load_models(args):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    yolo = YOLO(args.yolo_weights)
    yolo.fuse()
    yolo.to(device)

    tracker = DeepSort(max_iou_distance = 0.7, max_age = 1)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return yolo, tracker, transform, device

def draw(bbox, idx, dist, color, frame):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{idx}:{dist}%"
    cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    return frame
    


def preprocess_face(face, transform, device):
    face = Image.fromarray(face)
    face = transform(face).unsqueeze(0).to(device)
    return face

def run_inference(args):
    yolo, tracker, transform, device = load_models(args)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), "Cannot open video"

    const = 20

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = []
        
        results = yolo.predict(frame, conf=0.4, verbose=False)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            detections.append(([int(x1), int(y1), int(x2-x1), int(y2-y1)], float(score), int(class_id)))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            bbox = track.to_ltrb()
            idx = track.track_id

            x1, y1, x2, y2 = map(int, bbox)

            x1, y1 = max(0, x1 - const), max(0, y1 - const)
            x2, y2 = min(frame.shape[1], x2 + const), min(frame.shape[0], y2 + const)

            face = frame[y1:y2, x1:x2]

            face = preprocess_face(face, transform, device)

            emb = triplet.predict(face)
            dist = calculate_dist(orig_vector, emb)

            percent_dist = (1 - round(dist.item(), 2)) * 100

            if percent_dist > args.threshold:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            frame = draw(bbox, idx, percent_dist, color, frame)
        
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()




    

        