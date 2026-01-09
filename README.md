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


The dataset exhibits three major challenges related to imbalance and weak supervision:
1. Class Imbalance:
   The distribution of samples across classes is highly skewed, with normal activities
   dominating the dataset and certain crime categories having significantly fewer samples.
   It is handled using class-weighted loss and optional weighted sampling, and evaluated using balanced metrics (macro F1, balanced accuracy).

2. Video-Level and Temporal Imbalance:
   Crime-related events are sparse and short in duration, resulting in a limited number
   of frames depicting criminal activity compared to normal frames. Additionally, some
   classes are represented by a small number of videos, making video-wise stratified
   splitting essential for reliable evaluation. It is addressed through video-level stratified
   splitting and temporal uniform sampling during training.

3. Label Noise due to Weak Supervision:
   Annotations are provided at the video level, while training is performed at the frame
   level. Consequently, many frames labeled as crime do not visually contain the criminal
   activity, introducing label noise and increasing the difficulty of learning
   discriminative features. is mitigated by transfer learning and by computing video-level predictions through temporal aggregation during evaluation.


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
python -m src.main evaluate --ckpt experiments/mobilenetv2/debug_run/checkpoints/best_bal_acc.pt --model mobilenetv2 --agg_method mean_probs --smoothing ema_probs
```