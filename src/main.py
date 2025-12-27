from __future__ import annotations

import argparse

from src.train.train import TrainConfig, train
from src.evaluation.evaluate import evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crime Detection Pipeline")

    subparsers = p.add_subparsers(dest="command", required=True)

    # train
    train_p = subparsers.add_parser("train", help="Train a model")
    train_p.add_argument("--mode", type=str, default="debug", choices=["debug", "final"])

    # evaluate
    eval_p = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_p.add_argument("--ckpt", type=str, required=True)
    eval_p.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet50", "mobilenetv2", "vgg16"])
    eval_p.add_argument("--device", type=str, default="cpu")

    eval_p.add_argument("--root_dir", type=str, default="dataset")
    eval_p.add_argument("--out_dir", type=str, default="results")
    eval_p.add_argument("--batch_size", type=int, default=32)
    eval_p.add_argument("--num_workers", type=int, default=0)
    eval_p.add_argument("--img_size", type=int, default=224)
    eval_p.add_argument("--dropout", type=float, default=0.4)

    eval_p.add_argument("--max_per_class_test", type=int, default=None)

    eval_p.add_argument("--agg_method", type=str, default="mean_probs")
    eval_p.add_argument("--smoothing", type=str, default="none")
    eval_p.add_argument("--smoothing_alpha", type=float, default=0.7)

    return p.parse_args()


def main():
    args = parse_args()

    # train
    if args.command == "train":
        cfg = TrainConfig()
        cfg.data.num_workers = 0
        cfg.device = "cpu"

        if args.mode == "debug":
            cfg.data.max_per_class_train = 200
            cfg.data.max_per_class_test = 50
            cfg.data.weighted_sampling = False
            cfg.epochs = 3
            cfg.model_name = "mobilenetv2"
            cfg.experiment_name = "debug_run"

        elif args.mode == "final":
            cfg.data.max_per_class_train = None
            cfg.data.max_per_class_test = None
            cfg.data.weighted_sampling = True
            cfg.epochs = 25
            cfg.model_name = "resnet50"
            cfg.freeze_backbone = False
            cfg.experiment_name = "final_resnet50"

        out = train(cfg)
        print("\nTraining done:", out)

    # evaluate
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