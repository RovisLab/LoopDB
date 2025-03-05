# Command-Line Dataset Explorer for LoopDB

This command-line tool allows you to explore and validate the LoopDB dataset.

## Features

* Validate the dataset structure and integrity
* Display image sequences grouped by root image
* View metadata for specific images
* Open and display images

## Usage

1. Edit the `DATASET_PATH` variable in the script to point to your dataset directory
2. Run the script with one of the following commands:

### Validate the dataset
```
python cli_explorer.py validate
```

### View all image sequences
```
python cli_explorer.py sequences
```

### View metadata for a specific image
```
python cli_explorer.py view --metadata <image_timestamp>
```
Example:
```
python cli_explorer.py view --metadata IMG_20240525_163830
```

### View a specific image
```
python cli_explorer.py view --image <image_timestamp>
```
Example:
```
python cli_explorer.py view --image IMG_20240525_163830
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

