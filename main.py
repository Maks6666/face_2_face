from tracker import Tracker
from parser import parse_args

def main():
    args = parse_args()
    tracker = Tracker(path=args.video, yolo=args.yolo_weights, model_link=args.action_model, device=args.device, action_device=args.action_device, save=args.save_to_db)
    tracker()


if __name__ == '__main__':
    main()