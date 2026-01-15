import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Human faces tracker")
    parser.add_argument("--path", type=str, required=True, help="Path to videofile")
    parser.add_argument("--device", type=str, default="mps", help="mps | cpu")
    parser.add_argument("--yolo", type=str, default="./yolov12s-face.pt", help="YOLO face detector")
    parser.add_argument("--db", type=bool, default=False, help="Save to data base?")
    

    return parser.parse_args()