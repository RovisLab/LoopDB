# Feature Matching Tool for Image Sequences

This script finds matching keypoints between 5 images in a single scene and calculates transformations between consecutive images.

## Features

* Compares both SIFT and ORB feature detection methods
* Calculates homography transformations between consecutive images
* Decomposes transformations into rotation and translation components
* Visualizes keypoint matches between images
* Computes error metrics for transformations

## Processing Order

The script processes images in the sequence order:
`root_image -> image1 -> image2 -> image3 -> image4`

## Usage

1. Edit the `image_paths` list in the script to point to your 5 scene images
2. Run the script:
   ```
   python find_matches.py
   ```
3. View the visualizations and transformation results

## Output

* Visual display of feature matches between consecutive images
* Transformation matrices in the console
* Rotation and translation values for each image pair
* Error metrics for the transformations

## How It Works

1. Images are preprocessed to enhance feature detection
2. Features are detected using both SIFT and ORB algorithms
3. The better performing algorithm is selected for each image pair
4. Homography transformations are calculated from matched features
5. Transformations are decomposed into rotations and translations

