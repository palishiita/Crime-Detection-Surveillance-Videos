# Crime Detection in Surveillance Videos  
### Frame-Based Deep Learning Classification using VGG16, ResNet50, and MobileNetV2

This project implements a deep learning–based pipeline for classifying crime-related activities in CCTV surveillance footage.  
Instead of full video analysis, the system operates on **individual frames**, extracting visual features for six crime categories from the **UCF-Crime dataset**:

- **Abuse**
- **Assault**
- **Fighting**
- **Shooting**
- **Robbery**
- **Normal**

The goal is to evaluate different pretrained CNN architectures and compare their performance on frame-level crime classification.


## Commands

Debug Training Mode:
```bash
python -m src.main train --mode debug
```

Final Training Mode:
```bash
python -m src.main train --mode final
```

Evaluation Mode:
```bash
python -m src.main evaluate --ckpt experiments/resnet50/final_resnet50/checkpoints/best_bal_acc.pt --model resnet50 --agg_method mean_probs --smoothing ema_probs
```