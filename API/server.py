import csv
from flask import Flask, jsonify, request, send_file
import os
from PIL import Image
from io import BytesIO

app = Flask(__name__)

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
        descriptor_path_1 = os.path.join(self.datastream_1_path, "data_descriptor.csv")
        if os.path.exists(descriptor_path_1):
            with open(descriptor_path_1, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamp = row['timestamp_start']
                    if timestamp:  # Ensure timestamp is not None
                        self.metadata[timestamp] = {k: v if v is not None else "" for k, v in row.items()}

        descriptor_path_2 = os.path.join(self.datastream_2_path, "data_descriptor.csv")
        if os.path.exists(descriptor_path_2):
            with open(descriptor_path_2, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamp = row['timestamp_start']
                    timestamp_root = row['timestamp_root']
                    if timestamp in self.metadata:
                        self.metadata[timestamp].update(row)
                    else:
                        self.metadata[timestamp] = row
                    if timestamp_root not in self.root_images:
                        self.root_images[timestamp_root] = []
                    self.root_images[timestamp_root].append(timestamp)

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
        return None

    def get_metadata(self, timestamp):
        return self.metadata.get(timestamp)

    def get_image_sequences(self):
        return self.root_images

dataset = DatasetManager(DATASET_PATH)

@app.route('/validate', methods=['GET'])
def validate_dataset():
    response = {
        "datastream_1_exists": os.path.exists(dataset.datastream_1_path),
        "datastream_2_exists": os.path.exists(dataset.datastream_2_path),
        "sample_images_count": len(os.listdir(os.path.join(dataset.datastream_1_path, "samples", "left"))) if os.path.exists(dataset.datastream_1_path) else 0,
        "metadata_entries": len(dataset.metadata),
        "root_image_groups": len(dataset.root_images),
    }
    return jsonify(response)

@app.route('/view/image/<timestamp>', methods=['GET'])
def view_image(timestamp):
    img_path = dataset.get_image_path_for_viewing(timestamp)
    if img_path and os.path.exists(img_path):
        img = Image.open(img_path)
        
        # Resize the image to 800x600
        img = img.resize((800, 600))
        
        img_io = BytesIO()
        img.save(img_io, 'JPEG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')
    
    return jsonify({"error": "Image not found"}), 404

@app.route('/view/metadata/<timestamp>', methods=['GET'])
def view_metadata(timestamp):
    metadata = dataset.get_metadata(timestamp)
    
    if metadata:
        # Filter out None values in keys and values
        cleaned_metadata = {k: (v if v is not None else "") for k, v in metadata.items() if k is not None}

        # Debugging: Print the cleaned metadata to inspect it
        print(f"Cleaned Metadata for {timestamp}: {cleaned_metadata}")
        
        return jsonify(cleaned_metadata)
    
    return jsonify({"error": "Metadata not found"}), 404

@app.route('/sequences', methods=['GET'])
def view_sequences():
    sequences = dataset.get_image_sequences()
    return jsonify(sequences)

if __name__ == "__main__":
    app.run(debug=True)
