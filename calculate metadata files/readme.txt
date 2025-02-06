This script will calculate the transformation for each scene sequence, then will store the image name, the root image of the scene ,the transformation quaternion. Between the scene images in the next order:root_image -> image1-> image2-> image3-> image4.

to run the script ,you need to add the directory of the dataset .

the dataset should have the next structure:

.
└── Dataset/
    ├── datastream_1/
    │   ├── samples/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   ├── image3.jpg
    │   │   └── ..
    │   └── data_descriptor.csv
    └── datastream_2/
        └── data_descriptor.csv
