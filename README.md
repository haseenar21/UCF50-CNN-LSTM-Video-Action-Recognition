# UCF50 CNN-LSTM Video Action Recognition

A deep learning project for human action recognition from video sequences using a CNN-LSTM architecture.

The project uses a subset of the UCF50 action recognition dataset and demonstrates how spatial features can be extracted from individual video frames using a CNN and how temporal information across frames can be learned using an LSTM.

## Project Overview

Unlike image classification, video action recognition requires understanding both:

- Spatial information — what appears in each frame
- Temporal information — how the visual content changes across frames

To address this, the project uses a CNN to extract visual features from individual frames and an LSTM to learn temporal patterns across the sequence of frames.

## Dataset

The project uses 5 action classes from the UCF50 dataset:

- Basketball
- Biking
- Punch
- GolfSwing
- HorseRiding

Dataset distribution:

| Class | Training Videos | Test Videos |
|-------|------------------:|-------------:|
| Basketball | 109 | 28 |
| Biking | 116 | 29 |
| Punch | 128 | 32 |
| GolfSwing | 113 | 29 |
| HorseRiding | 157 | 40 |
| **Total** | **623** | **158** |

## Data Preprocessing

Each video is converted into a sequence of uniformly sampled frames.

Configuration used:

- Frames per video: 16
- Frame size: 112 × 112
- Channels: 3 (RGB)
- Input shape: `(16, 112, 112, 3)`
- Pixel values normalized to `[0, 1]`

Uniform frame sampling was used to capture information from different parts of each video rather than relying only on consecutive frames.

## Model Architecture

The project uses a CNN-LSTM architecture:

```text
Video
  ↓
16 Frames
  ↓
CNN Feature Extraction
  ↓
16 × 64 Feature Sequences
  ↓
LSTM (64 units)
  ↓
Dense Layer (32 units)
  ↓
Softmax Output
  ↓
5 Action Classes
