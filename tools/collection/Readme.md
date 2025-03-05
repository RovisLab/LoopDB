# scene Upload Tool for LoopDB Dataset

This tool provides a GUI interface to upload your own scene with image sequences to the LoopDB dataset.

## Features

* Upload a root image and its associated sequence images (4 additional images)
* Automatically organizes images into the dataset structure
* Creates necessary metadata entries in CSV files

## Usage

1. Edit the `base_path` variable in the script to point to your dataset directory
2. Run the script:
   ```
   python upload_images.py
   ```
3. Use the GUI to upload images:
   * First, select a root image
   * Then, select the 4 additional images in the sequence

## Post-Upload Processing

After uploading images, you need to run the transformation calculation script to compute the transformations between images:

```
python update_metadata.py
```

This will calculate transformations between images in the sequence order:
`root_image -> image1 -> image2 -> image3 -> image4`

## Required Directory Structure

The script will create this structure if it doesn't exist:

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

