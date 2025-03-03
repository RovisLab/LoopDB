import os
import csv
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

class ImageUploaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Image Uploader")

        # add the path for your dataset metadata folder ,  ex:r'C:\Users\dataset\metadata'
        self.base_path = r'C:\Users\dataset\metadata'
        
        self.datastream1_samples_left_path = os.path.join(self.base_path, "datastream_1", "samples", "left")
        os.makedirs(self.datastream1_samples_left_path, exist_ok=True)
        self.root_image = None
        
        self.label = tk.Label(root, text="Upload an image to the dataset:")
        self.label.pack(pady=10)
        
        self.upload_root_button = tk.Button(root, text="Upload Root Image", command=self.upload_root_image)
        self.upload_root_button.pack(pady=5)
        
        self.upload_subimages_button = tk.Button(root, text="Upload Subimages", command=self.upload_subimages, state=tk.DISABLED)
        self.upload_subimages_button.pack(pady=5)

    def update_csv(self, datastream_path, header, row):
        csv_path = os.path.join(datastream_path, "data_descriptor.csv")
        
        # Ensure CSV file exists and has the correct header
        if not os.path.exists(csv_path):
            with open(csv_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header)
        
        # Append the row to the CSV file without adding quotes around values
        with open(csv_path, mode='a', newline='') as csv_file:
            csv_file.write(','.join(map(str, row)) + '\n')

    def upload_root_image(self):
        file_path = filedialog.askopenfilename(title="Select a root image file", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if file_path and self.validate_image(file_path):
            self.root_image = file_path
            self.upload_subimages_button.config(state=tk.NORMAL)
            messagebox.showinfo("Success", "Root image uploaded successfully. Now you can upload subimages.")
        else:
            messagebox.showerror("Error", "Failed to upload root image. Please try again.")

    def upload_subimages(self):
        file_paths = filedialog.askopenfilenames(title="Select exactly 4 subimages", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if not file_paths or len(file_paths) != 4:
            messagebox.showerror("Error", "Please select exactly 4 subimages.")
            return
        
        for file_path in file_paths:
            if not self.validate_image(file_path):
                return
        
        root_filename = os.path.basename(self.root_image)
        root_file_name_without_ext = os.path.splitext(root_filename)[0]
        timestamp_root = root_file_name_without_ext
        rt = "rt_value"  # Placeholder, adjust as needed
        
        # Process the root image first
        self.process_image(self.root_image, is_root=True, timestamp_root=timestamp_root, rt=rt)
        
        # Process subimages
        for file_path in file_paths:
            self.process_image(file_path, is_root=False, timestamp_root=timestamp_root, rt=rt)

        messagebox.showinfo("Success", "Images uploaded and metadata updated successfully.")
        self.upload_subimages_button.config(state=tk.DISABLED)

    def validate_image(self, file_path):
        # Ensure the file is an image and does not already exist in the target directory
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension not in valid_extensions:
            messagebox.showerror("Error", f"Invalid file type: {file_extension}. Please select an image file.")
            return False
        
        filename = os.path.basename(file_path)
        destination_path = os.path.join(self.datastream1_samples_left_path, filename)
        if os.path.exists(destination_path):
            messagebox.showerror("Error", f"The file '{filename}' already exists in the target directory.")
            return False
        
        return True

    def process_image(self, file_path, is_root, timestamp_root, rt):
        filename = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(filename)[0]
        timestamp_start = file_name_without_ext
        timestamp_stop = file_name_without_ext
        sampling_time = 1
        left_file_path_0 = os.path.join("datastream_1", "samples", "left", file_name_without_ext)
        right_file_path_0 = "0, "
        
        # Copy the file to the samples/left folder
        destination_path = os.path.join(self.datastream1_samples_left_path, filename)
        shutil.copy(file_path, destination_path)
        
        # Update datastream_1 data_descriptor.csv
        datastream1_header = ["timestamp_start", "timestamp_stop", "sampling_time", "left_file_path_0", "right_file_path_0"]
        datastream1_row = [timestamp_start, timestamp_stop, sampling_time, left_file_path_0, right_file_path_0]
        self.update_csv(os.path.join(self.base_path, "datastream_1"), datastream1_header, datastream1_row)
        
        # Update datastream_2 data_descriptor.csv
        datastream2_header = ["timestamp_start", "timestamp_stop", "sampling_time", "timestamp_root"]
        datastream2_row = [timestamp_start, timestamp_stop, sampling_time, timestamp_root]
        self.update_csv(os.path.join(self.base_path, "datastream_2"), datastream2_header, datastream2_row)

# Create the main window
root = tk.Tk()
app = ImageUploaderApp(root)

# Start the main loop
root.mainloop()
