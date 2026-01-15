from tracker import Tracker
from parser import parse_args

def main():
    args = parse_args()
    tracker = Tracker(path=args.path, device=args.device, yolo_link=args.yolo, db=args.db)
    tracker()

if __name__ == "__main__":
    main()