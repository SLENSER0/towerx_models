import comet_ml
comet_ml.login(project_name="yolo")
from ultralytics import YOLO
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the YOLO model.")

    parser.add_argument('--comet_api_key', type=str, required=True)
    parser.add_argument('--comet_project_name', type=str, default="yolo")

    parser.add_argument('--model_path', type=str, default="yolo11x.pt")
    parser.add_argument('--data_path', type=str, default="data.yaml")

    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--imgsz', type=int, default=1520)
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['SGD', 'Adam', 'AdamW', 'RMSprop'])
    parser.add_argument('--cos_lr', action='store_true')
    parser.add_argument('--degrees', type=float, default=0.25)
    parser.add_argument('--scale', type=float, default=0.3)
    parser.add_argument('--save_period', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch', type=int, default=8)

    return parser.parse_args()

def main(args):
    comet_ml.login(api_key=args.comet_api_key)

    model = YOLO(args.model_path)
    train_params = {
        "data": args.data_path,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "device": args.device,
        "optimizer": args.optimizer,
        "cos_lr": args.cos_lr,
        "degrees": args.degrees,
        "scale": args.scale,
        "save_period": args.save_period,
        "workers": args.workers,
        "batch": args.batch
    }

    model.train(**train_params)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
