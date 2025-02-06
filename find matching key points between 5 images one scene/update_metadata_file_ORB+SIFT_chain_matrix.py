import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import time
from scipy.spatial.transform import Rotation as R

def preprocess_image(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def find_matches(detector, img1, img2):
    start_time = time.time()
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    print(f"    {detector.__class__.__name__} detectAndCompute time: {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    if detector.__class__.__name__ == 'SIFT':
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    elif detector.__class__.__name__ == 'ORB':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        good_matches = bf.match(des1, des2)
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:50]
    else:
        raise ValueError(f"Unsupported detector type: {detector.__class__.__name__}")
    
    print(f"    {detector.__class__.__name__} matching time: {time.time() - start_time:.4f} seconds")
    return kp1, kp2, good_matches

def draw_matches(img1, kp1, img2, kp2, matches, num_matches=50):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    
    out_img = np.zeros((max(rows1, rows2), cols1 + cols2, 3), dtype='uint8')
    out_img[:rows1, :cols1, :] = np.dstack([img1] * 3)
    out_img[:rows2, cols1:, :] = np.dstack([img2] * 3)
    
    for i, match in enumerate(matches[:num_matches]):
        (x1, y1) = kp1[match.queryIdx].pt
        (x2, y2) = kp2[match.trainIdx].pt
        
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        cv2.line(out_img, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), color, 5)
        cv2.circle(out_img, (int(x1), int(y1)), 3, color, 5)
        cv2.circle(out_img, (int(x2) + cols1, int(y2)), 3, color, 5)
    
    return out_img

def get_transformation(kp1, kp2, matches):
    if len(matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            return H
    return None

def decompose_homography(H):
    if H is None:
        return None
    
    _, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, np.eye(3))
    R = Rs[0]
    T = Ts[0]
    
    q, _ = cv2.Rodrigues(R)
    angle = np.linalg.norm(q)
    axis = q / angle if angle != 0 else q
    qw = np.cos(angle / 2.0)
    qx, qy, qz = axis * np.sin(angle / 2.0)
    
    return [float(qx), float(qy), float(qz), float(qw), float(T[0]), float(T[1]), float(T[2])]

def quaternion_translation_to_matrix(q, t):
    rot_matrix = R.from_quat(q).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = t
    return transform

def matrix_to_quaternion_translation(matrix):
    rot = R.from_matrix(matrix[:3, :3])
    q = rot.as_quat()
    t = matrix[:3, 3]
    return np.concatenate([q, t])

def process_image_pair(img1, img2, visualize=False):
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)

    sift = cv2.SIFT_create(nfeatures=2000)
    orb = cv2.ORB_create(nfeatures=2000)

    kp1_sift, kp2_sift, matches_sift = find_matches(sift, img1, img2)
    kp1_orb, kp2_orb, matches_orb = find_matches(orb, img1, img2)

    sift_match_count = len(matches_sift)
    orb_match_count = len(matches_orb)

    print(f"    SIFT Matches: {sift_match_count}")
    print(f"    ORB Matches: {orb_match_count}")

    if orb_match_count >= sift_match_count:
        best_method = 'ORB'
        kp1, kp2, matches = kp1_orb, kp2_orb, matches_orb
    else:
        best_method = 'SIFT'
        kp1, kp2, matches = kp1_sift, kp2_sift, matches_sift

    H = get_transformation(kp1, kp2, matches)
    transformation = decompose_homography(H)

    if visualize:
        out_img = draw_matches(img1, kp1, img2, kp2, matches, num_matches=70)
        plt.figure(figsize=(12, 10))
        plt.imshow(out_img)
        plt.title(f'Matches using {best_method}')
        plt.axis('off')
        plt.show()

    return transformation, best_method

def determine_root_image(index, rows):
    if rows[index]['timestamp_root']:
        return rows[index]['timestamp_root']
    root_index = (index // 5) * 5
    return rows[root_index]['timestamp_start']

def compute_cumulative_transformations(rows, images_path, visualize=False, visualize_interval=10):
    cumulative_transformations = []
    previous_transformation = np.eye(4)

    print(f"Total rows read from CSV: {len(rows)}")

    for i in range(1, len(rows)):
        print(f"\nProcessing pair {i}/{len(rows)-1}")

        current_image = rows[i-1]['timestamp_start']
        next_image = rows[i]['timestamp_start']
        root_image = determine_root_image(i, rows)

        print(f"    Current image: {current_image}")
        print(f"    Next image: {next_image}")
        print(f"    Root image: {root_image}")

        current_image_path = os.path.join(images_path, f"{current_image}.jpg")
        next_image_path = os.path.join(images_path, f"{next_image}.jpg")

        if not os.path.exists(current_image_path) or not os.path.exists(next_image_path):
            print(f"  Error: Image not found")
            cumulative_transformations.append(None)
            continue

        current_img = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
        next_img = cv2.imread(next_image_path, cv2.IMREAD_GRAYSCALE)

        if current_img is None or next_img is None:
            print(f"  Error: Could not read images")
            cumulative_transformations.append(None)
            continue

        try:
            transformation, best_method = process_image_pair(current_img, next_img, visualize=(visualize and i % visualize_interval == 0))

            if transformation:
                qx, qy, qz, qw, tx, ty, tz = transformation
                current_transformation = quaternion_translation_to_matrix([qx, qy, qz, qw], [tx, ty, tz])
                cumulative_transformation = np.dot(previous_transformation, current_transformation)
                previous_transformation = cumulative_transformation

                cumulative_transformations.append(cumulative_transformation)
                print(f"    Transformation calculated successfully using {best_method}")
            else:
                cumulative_transformations.append(None)
                print(f"    Error: No transformation found")
        except Exception as e:
            print(f"    Error processing images: {str(e)}")
            cumulative_transformations.append(None)
    
    return cumulative_transformations

def compare_with_root(cumulative_transformations, root_transformation):
    comparisons = []
    root_transformation_matrix = np.eye(4)  # Assuming root transformation is identity

    for cumulative_transformation in cumulative_transformations:
        if cumulative_transformation is None:
            comparisons.append(None)
        else:
            comparison = np.dot(np.linalg.inv(root_transformation_matrix), cumulative_transformation)
            comparisons.append(comparison)

    return comparisons

def main(visualize=False):
    base_path = r"C:\Users\maher\Desktop\PhD\Research\loop_closure_dataset\script\metadata"
    csv_path = os.path.join(base_path, "datastream_2", "data_descriptor.csv")
    images_path = os.path.join(base_path, "datastream_1", "samples", "left")

    print("Reading CSV file...")
    
    fieldnames = ['timestamp_start', 'timestamp_stop', 'sampling_time', 'timestamp_root', 
                  'q_1', 'q_2', 'q_3', 'q_w', 'tx', 'ty', 'tz']

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Remove any extra fields and strip whitespace, handling None values for both keys and values
            cleaned_row = {(k.strip() if k is not None else ''): (v.strip() if v is not None else '') 
                           for k, v in row.items() 
                           if k is not None and k.strip() in fieldnames}
            rows.append(cleaned_row)

    print(f"Processing {len(rows)} rows")

    cumulative_transformations = compute_cumulative_transformations(rows, images_path, visualize=visualize)

    root_transformation = np.eye(4)
    comparisons = compare_with_root(cumulative_transformations, root_transformation)

    for i, row in enumerate(rows):
        root_image = determine_root_image(i, rows)
        if i % 5 == 0:  # This is a root image
            row.update({
                'q_1': '0', 'q_2': '0', 'q_3': '0', 'q_w': '1',
                'tx': '0', 'ty': '0', 'tz': '0',
                'timestamp_root': root_image
            })
        else:
            transformation = comparisons[i-1] if i > 0 else None
            if transformation is None:
                row.update({
                    'q_1': '', 'q_2': '', 'q_3': '', 'q_w': '',
                    'tx': '', 'ty': '', 'tz': '',
                    'timestamp_root': root_image
                })
            else:
                q_t = matrix_to_quaternion_translation(transformation)
                row.update({
                    'q_1': f"{q_t[0]:.6f}", 'q_2': f"{q_t[1]:.6f}", 'q_3': f"{q_t[2]:.6f}", 'q_w': f"{q_t[3]:.6f}",
                    'tx': f"{q_t[4]:.6f}", 'ty': f"{q_t[5]:.6f}", 'tz': f"{q_t[6]:.6f}",
                    'timestamp_root': root_image
                })
        
        print(f"    Image: {row['timestamp_start']}, Root image: {row['timestamp_root']}")

    updated_csv_path = os.path.join(base_path, "updated_data_descriptor.csv")
    with open(updated_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV file updated and saved as {updated_csv_path}")

if __name__ == "__main__":
    main(visualize=False)