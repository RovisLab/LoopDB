# Transformation Calculator for LoopDB Dataset

This script calculates transformations between consecutive images in each scene sequence of the LoopDB dataset.

## Features

* Automatically detects features using both SIFT and ORB algorithms
* Computes homography transformations between consecutive images
* Decomposes transformations into quaternion rotations and translations
* Creates cumulative transformations relative to the root image
* Stores results in the dataset metadata file
* Optional visualization of matches and transformations

## Calculation Process

The script processes images in sequence order:
`root_image -> image1 -> image2 -> image3 -> image4`

For each pair of consecutive images, it:
1. Detects and matches features
2. Computes the homography transformation
3. Decomposes into quaternion rotation and translation
4. Builds cumulative transformations
5. Updates the metadata CSV file

## Usage

1. Edit the `base_path` variable in the script to point to your dataset directory
2. Run the script:
   ```
   python update_metadata.py
   ```
3. Optional: Enable visualizations with the `visualize=True` parameter:
   ```python
   main(base_path, visualize=True)
   ```

## Required Directory Structure

```
dataset/
├── datastream_1/
│   ├── samples/
│   │   └── left/
│   │       ├── image1.jpg
│   │       ├── image2.jpg
│   │       └── ...
│   └── data_descriptor.csv
└── datastream_2/
    └── data_descriptor.csv
```

## Output

The script updates `datastream_2/data_descriptor.csv` with:
* Image timestamps
* Root image reference
* Quaternion rotation values (q_1, q_2, q_3, q_w)
* Translation values (tx, ty, tz)

