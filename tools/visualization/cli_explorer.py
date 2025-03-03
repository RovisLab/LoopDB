import os
import csv
import argparse
from PIL import Image

# add the path for your dataset metadata folder ,  ex:r'C:\Users\dataset\metadata'
DATASET_PATH = r'C:\Users\dataset\metadata'
# -----------------------------------------------------------------

class DatasetManager:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.datastream_1_path = os.path.join(dataset_path, "datastream_1")
        self.datastream_2_path = os.path.join(dataset_path, "datastream_2")
        self.metadata = {}
        self.root_images = {}
        self.load_data()

    def load_data(self):
        # Load metadata and image paths from datastream_1 (for image paths)
        descriptor_path_1 = os.path.join(self.datastream_1_path, "data_descriptor.csv")
        if os.path.exists(descriptor_path_1):
            with open(descriptor_path_1, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamp = row['timestamp_start']
                    self.metadata[timestamp] = row  # This includes 'left_file_path_0'
        
        # Load additional metadata from datastream_2 (if necessary)
        descriptor_path_2 = os.path.join(self.datastream_2_path, "data_descriptor.csv")
        if os.path.exists(descriptor_path_2):
            with open(descriptor_path_2, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamp = row['timestamp_start']
                    timestamp_root = row['timestamp_root']
                    
                    # Add or update metadata with data from datastream_2
                    if timestamp in self.metadata:
                        self.metadata[timestamp].update(row)
                    else:
                        self.metadata[timestamp] = row
                    
                    # Group images by their root image
                    if timestamp_root not in self.root_images:
                        self.root_images[timestamp_root] = []
                    self.root_images[timestamp_root].append(timestamp)
        else:
            print(f"Data descriptor file not found: {descriptor_path_2}")

    def view_image(self, timestamp):
        img_path = self.get_image_path_for_viewing(timestamp)
        if img_path:
            print(f"Attempting to open image at: {img_path}")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img.show()
            else:
                print(f"Image not found at: {img_path}")
                print(f"Directory contents: {os.listdir(os.path.dirname(img_path))}")
        else:
            print(f"Image path could not be determined for timestamp: {timestamp}")

    def get_image_path(self, timestamp):
        # Retrieve the image path from the metadata using the left_file_path_0 column
        metadata = self.metadata.get(timestamp)
        if metadata:
            relative_path = metadata.get('left_file_path_0')
            if relative_path:
                # Return the relative path directly from the CSV
                return relative_path
            else:
                print("Error: 'left_file_path_0' column not found in metadata.")
        return None

    def get_image_path_for_viewing(self, timestamp):
        metadata = self.metadata.get(timestamp)
        if metadata:
            relative_path = metadata.get('left_file_path_0')
            if relative_path:
                image_dir = os.path.join(self.datastream_1_path, "samples", "left")
                full_path = os.path.join(image_dir, os.path.basename(relative_path))
                if not full_path.lower().endswith('.jpg'):
                    full_path += '.jpg'
                return full_path
            else:
                print("Error: 'left_file_path_0' column not found in metadata.")
        return None

    def view_metadata(self, timestamp):
        metadata = self.metadata.get(timestamp)
        if metadata:
            print(f"Metadata for timestamp {timestamp}:")
            for key, value in metadata.items():
                print(f"{key}: {value}")

            # Show the image path using the correct relative path from the CSV
            img_path = self.get_image_path_for_viewing(timestamp)
            if img_path:
                print(f"Image Path: {img_path}")

            # Show the sequence of images for the root image
            root_image = metadata['timestamp_root']
            sequence = self.root_images.get(root_image, [])
            if sequence:
                print("\nSequence of images for root image:")
                for seq_img in sequence:
                    print(f"  - {seq_img}")

        else:
            print(f"Metadata not found for timestamp: {timestamp}")

    def validate_dataset(self):
        print("Validating entire dataset...")

        # Validate datastream 1
        print("Validating datastream 1...")
        if os.path.exists(self.datastream_1_path):
            print(f"Datastream 1 path exists: {self.datastream_1_path}")
            samples_path = os.path.join(self.datastream_1_path, "samples", "left")
            if os.path.exists(samples_path):
                print(f"Samples directory exists: {samples_path}")
                image_count = len(os.listdir(samples_path))
                print(f"Number of images: {image_count}")
            else:
                print(f"Samples directory not found: {samples_path}")
        else:
            print(f"Datastream 1 path not found: {self.datastream_1_path}")

        # Validate datastream 2
        print("\nValidating datastream 2...")
        if os.path.exists(self.datastream_2_path):
            print(f"Datastream 2 path exists: {self.datastream_2_path}")
            descriptor_path = os.path.join(self.datastream_2_path, "data_descriptor.csv")
            if os.path.exists(descriptor_path):
                print(f"Data descriptor file exists: {descriptor_path}")
                with open(descriptor_path, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    print(f"CSV header: {header}")
                    row_count = sum(1 for row in reader)
                    print(f"Number of data entries: {row_count}")
            else:
                print(f"Data descriptor file not found: {descriptor_path}")
        else:
            print(f"Datastream 2 path not found: {self.datastream_2_path}")

        # Check for consistency between images and metadata
        missing_images = set(self.metadata.keys()) - set(self.metadata.keys())
        missing_metadata = set(self.metadata.keys()) - set(self.metadata.keys())

        if missing_images:
            print("\nImages missing metadata:")
            for timestamp in missing_images:
                print(timestamp)

        if missing_metadata:
            print("\nMetadata missing images:")
            for timestamp in missing_metadata:
                print(timestamp)

        if not missing_images and not missing_metadata:
            print("\nDataset is valid. All images have corresponding metadata.")

    def display_image_sequences(self):
        """Displays image sequences grouped by their root image, excluding the root image itself."""
        print("Displaying image sequences grouped by root image:\n")
        for root_image, timestamps in self.root_images.items():
            print(f"Root Image: {root_image}")
            for timestamp in timestamps:
                if timestamp != root_image:
                    print(f"  - {timestamp}")
            print()  # Blank line for separation

def main():
    parser = argparse.ArgumentParser(description="Dataset Management CLI")
    parser.add_argument("--dataset-path", default=DATASET_PATH, help="Specify the root path to the dataset folder.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for the 'validate' command
    subparsers.add_parser('validate', help="Validate the dataset integrity.")

    # Subparser for the 'view' command
    parser_view = subparsers.add_parser('view', help="View image or metadata.")
    parser_view.add_argument("--image", metavar="TIMESTAMP", help="View image for the given timestamp")
    parser_view.add_argument("--metadata", metavar="TIMESTAMP", help="View metadata for the given timestamp")

    # Subparser for the 'sequences' command
    subparsers.add_parser('sequences', help="Display image sequences grouped by root image.")

    args = parser.parse_args()

    dataset = DatasetManager(args.dataset_path)

    if args.command == 'validate':
        dataset.validate_dataset()
    elif args.command == 'view':
        if args.image:
            dataset.view_image(args.image)
        elif args.metadata:
            dataset.view_metadata(args.metadata)
    elif args.command == 'sequences':
        dataset.display_image_sequences()

if __name__ == "__main__":
    main()
