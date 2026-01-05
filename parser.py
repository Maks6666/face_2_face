import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Hunams actions tracking")

    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--yolo_weights", type=str, default="./yolo12s.pt", help="YOLO weights")
    parser.add_argument("--action_model", type=str, default="./student_model.pt", help="Action classification model weights")
    parser.add_argument("--device", type=str, default="mps", help="cpu | mps | gpu")
    parser.add_argument("--action_device", type=str, default="cpu", help="cpu | mps | gpu"), 
    parser.add_argument("--save_to_db", type=bool, default=False, help="True | False")

    return parser.parse_args()
