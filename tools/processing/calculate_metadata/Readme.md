# Transformation Calculator for LoopDB Dataset

This script calculates transformations between consecutive images in each scene sequence of the LoopDB dataset.

## Features

* Automatically detects features using both SIFT and ORB algorithms
* Computes homography transformations between consecutive images
* Decomposes transformations into quaternion rotations and translations
* Creates cumulative transformations relative to the root image
* Stores results in the dataset metadata file
* Optional visualization of matches and transformations
*Visualization**: Optional visualization of feature matches and transformations.
*Error Projection**: Analyzes projection errors between computed and ground truth transformations.

## Calculation Process

The script processes images in sequence order:
`root_image -> image1 -> image2 -> image3 -> image4`

For each pair of consecutive images, it:
1. Detects and matches features
2. Computes the homography transformation
3. Decomposes into quaternion rotation and translation
4. Builds cumulative transformations
5. Updates the metadata CSV file

## Visualization

The script can generate visualizations of:
- **Feature Matches**: Displays matching keypoints between consecutive images.
- **Projection Errors**: Visualizes the difference between computed and ground truth projections.
- **Combined Scene Visualizations**: Creates a combined visualization of all images in a scene with their transformations.

Visualizations are saved as high-resolution images in the output directory.

## Error Projection

The script calculates and analyzes projection errors to evaluate the accuracy of computed transformations. It provides:
- **Error Metrics**: Mean, median, and maximum projection errors.
- **Error Distribution**: A histogram of projection errors for each feature detection method (SIFT and ORB).
- **Visualizations**: Visual representations of projection errors for each scene.


## Usage

1. Edit the `base_path` variable in the script to point to your dataset directory
2. Run the script:
   ```
   python loopdb_calculation.py
   ```
3. Optional: Enable visualizations with the `visualize=True` parameter:
   ```python
   main(base_path, visualize=True)
   ```
4. show Visualization and Error Projection :

python loopdb_calculation.py --path /path/to/dataset/metadata --visualize --error-projection
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

Visualization Output:
* Feature Matches: Saved as transform_{image1}_to_{image2}_matches.jpg.

Projection Errors: 
* Saved as: scene{scene_id}_{detector_name}_projection_error.jpg.

* Error Metrics: 
Saved in projection_error_results.csv

* Summary Statistics: 
Saved in projection_error_summary.csv.
