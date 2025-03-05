import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from scipy.spatial.transform import Rotation as R

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

def find_matches(detector, img1, img2):
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return kp1, kp2, []

    if detector.__class__.__name__ == 'SIFT':
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    elif detector.__class__.__name__ == 'ORB':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        good_matches = sorted(matches, key=lambda x: x.distance)[:50]
    else:
        raise ValueError(f"Unsupported detector type: {detector.__class__.__name__}")

    return kp1, kp2, good_matches

def compute_inliers(kp1, kp2, matches):
    if len(matches) < 4:
        return 0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return 0

    return np.sum(mask)

def compare_detectors(img1, img2):
    sift = cv2.SIFT_create(nfeatures=2000)
    kp1_sift, kp2_sift, matches_sift = find_matches(sift, img1, img2)
    inliers_sift = compute_inliers(kp1_sift, kp2_sift, matches_sift)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1_orb, kp2_orb, matches_orb = find_matches(orb, img1, img2)
    inliers_orb = compute_inliers(kp1_orb, kp2_orb, matches_orb)

    inliers_dict = {
        'sift': (kp1_sift, kp2_sift, matches_sift, inliers_sift),
        'orb': (kp1_orb, kp2_orb, matches_orb, inliers_orb)
    }
    
    best_detector = max(inliers_dict, key=lambda k: inliers_dict[k][3])
    return best_detector, inliers_dict[best_detector]

def get_transformation(kp1, kp2, matches):
    if len(matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
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

def preprocess_image(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def main():
    # add the path for the scene sequence ,  ex:r'C:\Users\IMG_20240525_171754.jpg'
    image_paths = [
        r'C:\Users\barak\Desktop\PhD\Research\loop_closure_dataset\dataset\0019\0.jpg',
        r'C:\Users\barak\Desktop\PhD\Research\loop_closure_dataset\dataset\0019\1.jpg',
        r'C:\Users\barak\Desktop\PhD\Research\loop_closure_dataset\dataset\0019\2.jpg',
        r'C:\Users\barak\Desktop\PhD\Research\loop_closure_dataset\dataset\0019\3.jpg',
        r'C:\Users\barak\Desktop\PhD\Research\loop_closure_dataset\dataset\0019\4.jpg'
    ]

    consecutive_matrices = []
    chained_matrices = [np.eye(4)]

    for i in range(len(image_paths) - 1):
        img1 = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_paths[i+1], cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            print(f"Error: Unable to read images at {image_paths[i]} or {image_paths[i+1]}")
            continue

        img1 = preprocess_image(img1)
        img2 = preprocess_image(img2)

        best_detector, (kp1, kp2, matches, inliers) = compare_detectors(img1, img2)
        print(f"Image {i} to Image {i+1} using {best_detector.upper()}: {len(matches)} matches, {inliers} inliers found")
        
        H_consecutive = get_transformation(kp1, kp2, matches)
        consecutive_transformation = decompose_homography(H_consecutive)
        
        if consecutive_transformation:
            consecutive_matrix = quaternion_translation_to_matrix(consecutive_transformation[:4], consecutive_transformation[4:])
            consecutive_matrices.append((best_detector, consecutive_matrix))
            chained_matrices.append(chained_matrices[-1] @ consecutive_matrix)
        
        print(f"Consecutive transformation from Image {i} to Image {i+1}: {consecutive_transformation}")

        # Visualization of consecutive matches
        out_img = draw_matches(img1, kp1, img2, kp2, matches, num_matches=70)
        plt.figure(figsize=(12, 10))
        plt.imshow(out_img)
        plt.title(f'Image {i} to Image {i+1} using {best_detector.upper()}')
        plt.axis('off')
        plt.show()

    # Calculate errors and prepare results
    results = []
    for i in range(1, len(chained_matrices)):
        chained_matrix = chained_matrices[i]
        prev_chained_matrix = chained_matrices[i-1]
        
        relative_matrix = np.linalg.inv(prev_chained_matrix) @ chained_matrix
        
        translation_error = np.linalg.norm(relative_matrix[:3, 3])
        rotation_error = np.degrees(np.linalg.norm(R.from_matrix(relative_matrix[:3, :3]).as_rotvec()))
        
        results.append([f"Image {i-1} to Image {i}", translation_error, rotation_error])
        print(f"Calculation for Image {i-1} to Image {i}: Translation: {translation_error}, Rotation: {rotation_error}°")

    # Write Results to CSV
    with open('data_descriptor_sift+orb_sequential.csv', 'w', newline='') as csvfile:
        fieldnames = ['Pair', 'Translation Error', 'Rotation Error (degrees)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow({
                'Pair': result[0],
                'Translation Error': result[1],
                'Rotation Error (degrees)': result[2]
            })

if __name__ == '__main__':
    main()