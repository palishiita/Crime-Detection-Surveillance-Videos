from __future__ import annotations
import argparse
import torch
from src.train.train import TrainConfig, train
from src.evaluation.evaluate import evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crime Detection Pipeline")
    subparsers = p.add_subparsers(dest="command", required=True)
    train_p = subparsers.add_parser("train", help="Train a model")
    train_p.add_argument("--mode", type=str, default="debug", choices=["debug", "final"])
    eval_p = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_p.add_argument("--ckpt", type=str, required=True)
    eval_p.add_argument("--model", type=str, default="mobilenetv2", choices=["resnet50", "mobilenetv2", "vgg16"])
    eval_p.add_argument("--device", type=str, default="cpu")
    eval_p.add_argument("--root_dir", type=str, default="dataset")
    eval_p.add_argument("--out_dir", type=str, default="results")
    eval_p.add_argument("--batch_size", type=int, default=32)
    eval_p.add_argument("--num_workers", type=int, default=0)
    eval_p.add_argument("--img_size", type=int, default=224)
    eval_p.add_argument("--dropout", type=float, default=0.4)
    eval_p.add_argument("--max_per_class_test", type=int, default=None)
    eval_p.add_argument("--agg_method", type=str, default="topk_mean_probs:0.05")
    eval_p.add_argument("--smoothing", type=str, default="none")
    eval_p.add_argument("--smoothing_alpha", type=float, default=0.7)
    return p.parse_args()


def main():
    args = parse_args()

    if args.command == "train":
        cfg = TrainConfig()
        cfg.data.num_workers = 0
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.data.return_meta = True
        cfg.data.val_split = 0.1
        cfg.data.video_wise_split = True

        if args.mode == "debug":
            cfg.data.max_per_class_train = 100
            cfg.data.max_per_class_test = 10
            cfg.data.weighted_sampling = False
            cfg.epochs = 3
            cfg.model_name = "mobilenetv2"
            cfg.experiment_name = "debug_run"
            cfg.video_agg_method = "topk_mean_probs:0.10"
            cfg.video_smoothing = "none"
            cfg.monitor_metric = "video_macro_f1"

        elif args.mode == "final":
            cfg.data.max_per_class_train = 7000
            cfg.data.max_per_class_test = 1500
            cfg.data.weighted_sampling = False
            cfg.epochs = 25
            cfg.model_name = "mobilenetv2"
            cfg.freeze_backbone = True
            cfg.fine_tune = True
            cfg.fine_tune_start_epoch = 4
            cfg.head_lr = 1e-4
            cfg.backbone_lr = 1e-5
            cfg.video_agg_method = "topk_mean_probs:0.05"
            cfg.video_smoothing = "none"
            cfg.monitor_metric = "video_macro_recall"
            cfg.experiment_name = "final_mobilenet_topk"

        out = train(cfg)
        print("\nTraining done:", out)

    elif args.command == "evaluate":
        outputs = evaluate(
            ckpt_path=args.ckpt,
            model_name=args.model,
            root_dir=args.root_dir,
            out_dir=args.out_dir,
            device_str=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            dropout=args.dropout,
            agg_method=args.agg_method,
            smoothing=args.smoothing,
            smoothing_alpha=args.smoothing_alpha,
            max_per_class_test=args.max_per_class_test,
        )
        print("\nEvaluation outputs:", outputs)


if __name__ == "__main__":
    main()