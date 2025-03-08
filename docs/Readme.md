# LoopDB Dataset Format

This document details the structure and format of the LoopDB dataset, designed for loop closure detection and SLAM algorithms.

## Overview

LoopDB consists of sequences of images organized into "submaps", where each submap contains 5 consecutive images of the same scene from different viewpoints. The dataset provides detailed metadata, including transformations between consecutive images.

## Data Organization

The dataset is organized into two main datastreams:

### Datastream 1

Contains the captured scene images and basic metadata.

```
datastream_1/
├── samples/
│   └── left/        # Image files (.jpg)
└── data_descriptor.csv
```

#### Datastream 1 CSV Format

The `data_descriptor.csv` file contains the following columns:

| Column | Description |
|--------|-------------|
| `timestamp_start` | Timestamp when image capture started (used as image ID) |
| `timestamp_stop` | Timestamp when image capture ended |
| `sampling_time` | Time spent capturing the image (in seconds) |
| `left_file_path_0` | Path to the image file within the dataset |
| `right_file_path_0` | Path to right camera image (unused in current version - monocamera only) |

### Datastream 2

Contains transformation data between images.

```
datastream_2/
└── data_descriptor.csv
```

#### Datastream 2 CSV Format

The `data_descriptor.csv` file contains the following columns:

| Column | Description |
|--------|-------------|
| `timestamp_start` | Timestamp/ID of the image |
| `timestamp_stop` | Timestamp when image capture ended |
| `sampling_time` | Number of times the same image was captured |
| `timestamp_root` | ID of the root image of this sequence |
| `q_1` | X component of quaternion rotation |
| `q_2` | Y component of quaternion rotation |
| `q_3` | Z component of quaternion rotation |
| `q_w` | W component of quaternion rotation |
| `tx` | X component of translation vector |
| `ty` | Y component of translation vector |
| `tz` | Z component of translation vector |

## Transformations

### Coordinate System

LoopDB uses a standard right-handed coordinate system commonly used in computer vision and robotics.

### Homography Transformation

The transformation between two images is represented by a homography matrix H, which encapsulates both rotation and translation:

H = [R | t]
    [0 | 1]

Where:
- R is the 3×3 rotation matrix
- t is the 3×1 translation vector

### Quaternion Representation

Instead of directly storing rotation matrices, LoopDB uses quaternions (q_1, q_2, q_3, q_w) to represent rotations, offering a compact representation that avoids gimbal lock issues.

## Submaps

Each scene in LoopDB is captured as a "submap" consisting of 5 images:

1. **Root Image**: First image in the sequence, serving as the reference frame
   - Has rotation [0,0,0,1] (identity quaternion) and translation [0,0,0]
   - All other images in the sequence are transformed relative to this root image

2. **Sequence Images**: Four additional images captured from different viewpoints
   - Each has transformation data relative to the root image

## Image Properties

- **Resolution**: 48MP (Huawei Nova 7i camera)
- **Format**: JPEG
- **Variability**: Images were taken with different ISO settings and exposure times

