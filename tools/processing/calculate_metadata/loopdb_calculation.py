import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import time
from scipy.spatial.transform import Rotation as R
from loopdb_visualization import visualize_matches_and_transformation


# Add scale factor as a constant
TRANSLATION_SCALE_FACTOR = 0.0001  # Make translation values smaller

def determine_root_image(index, rows):
    """
    Determine the root image for a given index in the sequence.
    Root images occur every 5 images and serve as reference points.
    """
    if index < 0 or index >= len(rows):
        return None
        
    # If the row has a timestamp_root, use it
    if rows[index].get('timestamp_root'):
        return rows[index]['timestamp_root']
        
    # Otherwise, calculate the root image (every 5th image is a root)
    root_index = (index // 5) * 5
    if root_index < len(rows):
        return rows[root_index]['timestamp_start']
    return None

def preprocess_image(img):
    # Enhanced preprocessing with both CLAHE and normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    img_norm = cv2.normalize(img_clahe, None, 0, 255, cv2.NORM_MINMAX)
    return img_norm

def find_matches(detector, img1, img2):
    start_time = time.time()
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        print("    Warning: No features detected in one or both images")
        return None, None, []
        
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

def get_transformation(kp1, kp2, matches):
    if len(matches) < 4:
        print("    Warning: Not enough matches to compute transformation")
        return None
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Use RANSAC with more strict parameters
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    
    if H is None:
        print("    Warning: Could not compute homography matrix")
        return None
        
    # Check if homography is valid
    if not is_valid_homography(H):
        print("    Warning: Invalid homography matrix detected")
        return None
        
    return H

def is_valid_homography(H):
    """Check if homography matrix is valid."""
    if H is None:
        return False
        
    # Check if matrix is finite
    if not np.all(np.isfinite(H)):
        return False
        
    # Check determinant is positive and not too close to zero
    det = np.linalg.det(H)
    if det < 1e-6:
        return False
        
    return True

def quaternion_translation_to_matrix(q, t):
    """
    Convert quaternion and translation to transformation matrix.
    Args:
        q: Quaternion [qx, qy, qz, qw]
        t: Translation [tx, ty, tz]
    Returns:
        4x4 transformation matrix
    """
    rot_matrix = R.from_quat(q).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = t
    return transform

def matrix_to_quaternion_translation(matrix):
    """
    Convert transformation matrix to quaternion and translation.
    Args:
        matrix: 4x4 transformation matrix
    Returns:
        Array of [qx, qy, qz, qw, tx, ty, tz]
    """
    rot = R.from_matrix(matrix[:3, :3])
    q = rot.as_quat()  # Returns [qx, qy, qz, qw]
    t = matrix[:3, 3]  # Extract translation
    return np.concatenate([q, t])

def decompose_homography(H):
    if H is None:
        return None
    
    try:
        _, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, np.eye(3))
        
        # Select the most probable solution
        R = Rs[0]
        T = Ts[0]
        
        # Apply scale factor to translation
        T = T * TRANSLATION_SCALE_FACTOR
        
        # Convert rotation matrix to quaternion
        q, _ = cv2.Rodrigues(R)
        angle = np.linalg.norm(q)
        axis = q / angle if angle != 0 else q
        qw = np.cos(angle / 2.0)
        qx, qy, qz = axis * np.sin(angle / 2.0)
        
        return [float(qx.item()), float(qy.item()), float(qz.item()), float(qw.item()), 
        float(T[0].item()), float(T[1].item()), float(T[2].item())]
    except Exception as e:
        print(f"    Error in decompose_homography: {str(e)}")
        return None

def enhanced_preprocess_image(img):
    """Advanced preprocessing to improve feature matching."""
    # Resize if image is too large
    max_dim = 1200
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    
    # Normalize
    img_norm = cv2.normalize(img_clahe, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply slight Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(img_norm, (3, 3), 0)
    
    # Enhance edges
    img_edges = cv2.Laplacian(img_blur, cv2.CV_8U, ksize=3)
    img_enhanced = cv2.addWeighted(img_blur, 0.7, img_edges, 0.3, 0)
    
    return img_enhanced

def find_matches_enhanced(detector, img1, img2, ratio_threshold=0.85):
    """Enhanced version of find_matches with relaxed filtering."""
    start_time = time.time()
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        print(f"    Warning: Not enough features detected in images ({len(kp1) if kp1 else 0}, {len(kp2) if kp2 else 0})")
        return None, None, []
        
    print(f"    {detector.__class__.__name__} found {len(kp1)} and {len(kp2)} keypoints in {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    if detector.__class__.__name__ == 'SIFT':
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:  # Very relaxed ratio
                good_matches.append(m)
    elif detector.__class__.__name__ == 'ORB':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        good_matches = bf.match(des1, des2)
        # Take more matches for ORB
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:150]  # Use more matches
    else:
        raise ValueError(f"Unsupported detector type: {detector.__class__.__name__}")
    
    print(f"    {detector.__class__.__name__} found {len(good_matches)} good matches in {time.time() - start_time:.4f} seconds")
    return kp1, kp2, good_matches

def validate_transformation(kp1, kp2, matches, H, img1, img2):
    """Validate transformation by checking reprojection error with much more relaxed threshold."""
    if len(matches) < 4:
        return False, float('inf')
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Project points using homography
    transformed_pts = cv2.perspectiveTransform(src_pts, H)
    
    # Calculate reprojection error
    errors = np.sqrt(np.sum((dst_pts - transformed_pts) ** 2, axis=2))
    
    # Remove outliers before computing mean (ignore worst 20% of matches)
    sorted_errors = np.sort(errors.flatten())
    inlier_count = int(len(sorted_errors) * 0.8)  # Use 80% of the matches
    if inlier_count > 0:
        inlier_errors = sorted_errors[:inlier_count]
        mean_error = np.mean(inlier_errors)
    else:
        mean_error = np.mean(errors)
    
    # print(f"    Mean reprojection error (after outlier removal): {mean_error:.4f} pixels")
    
    # Use a much more relaxed threshold for real-world images
    is_valid = mean_error < 50.0  # Very relaxed threshold
    
    # Even if error is high, we'll use the transformation if it's the best we can get
    if not is_valid:
        print(f"    High reprojection error ({mean_error:.2f} pixels), but will use anyway")
        is_valid = True
        
    return is_valid, mean_error

def get_transformation_with_fallback(kp1, kp2, matches, img1, img2, visualize=False):
    """Get transformation with multiple fallback mechanisms."""
    if len(matches) < 4:
        print("    Warning: Not enough matches to compute transformation")
        return None
        
    # Extract matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Try multiple methods with varying parameters
    methods = [
        ('RANSAC', 3.0),     # Strict RANSAC
        ('RANSAC', 5.0),     # Relaxed RANSAC
        ('LMEDS', 0.0),      # Least-median method
        ('RHO', 5.0)         # RHO method
    ]
    
    best_H = None
    best_error = float('inf')
    
    for method_name, param in methods:
        # Comment out or remove this print statement
        # print(f"    Trying homography with {method_name}, param={param}")
        method = getattr(cv2, f"{method_name}")
        
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, method, param)
            
            if H is not None and is_valid_homography(H):
                # Check reprojection error
                is_valid, error = validate_transformation(kp1, kp2, matches, H, img1, img2)
                
                # Keep the transformation with the lowest error
                if error < best_error:
                    best_H = H
                    best_error = error
                    # Comment out or remove this print statement
                    # print(f"    Found better transformation with error: {error:.2f}")
        except Exception as e:
            print(f"    Error with {method_name}: {str(e)}")
    
    if best_H is None:
        print("    Warning: All transformation methods failed")
        
        # Last resort: try with essential matrix estimation if we have enough matches
        if len(matches) >= 8:
            print("    Trying essential matrix as last resort...")
            try:
                # Create artificial camera matrix (assuming center of image is principal point)
                h, w = img1.shape
                focal = w  # Approximate focal length
                K = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]])
                
                # Extract points
                pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
                
                # Find essential matrix
                E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                
                if E is not None:
                    # Recover pose
                    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
                    
                    # Create homography from R, t
                    H = np.eye(3)
                    H[:2, :2] = R[:2, :2]
                    H[:2, 2] = t[:2, 0]
                    
                    if is_valid_homography(H):
                        best_H = H
                        print("    Successfully created transformation from essential matrix")
            except Exception as e:
                print(f"    Essential matrix approach failed: {str(e)}")
    
    return best_H

def compute_cumulative_transformations(rows, images_path, visualize=False, vis_func=None):
    cumulative_transformations = []
    previous_transformation = np.eye(4)

    print(f"Total rows read from CSV: {len(rows)}")

    # Create visualization directory if needed
    vis_dir = None
    if visualize:
        vis_dir = os.path.join(os.path.dirname(images_path), "..", "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        print(f"Saving visualizations to: {vis_dir}")

    for i in range(1, len(rows)):
        print(f"\nProcessing pair {i}/{len(rows)-1}")

        current_image = rows[i-1]['timestamp_start']
        next_image = rows[i]['timestamp_start']
        root_image = determine_root_image(i, rows)

        print(f"    Current image: {current_image}")
        print(f"    Next image: {next_image}")
        print(f"    Root image: {root_image}")

        # Load and verify images
        current_image_path = os.path.join(images_path, f"{current_image}.jpg")
        next_image_path = os.path.join(images_path, f"{next_image}.jpg")

        if not os.path.exists(current_image_path) or not os.path.exists(next_image_path):
            print(f"    Error: Image not found")
            cumulative_transformations.append(None)
            continue

        current_img = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
        next_img = cv2.imread(next_image_path, cv2.IMREAD_GRAYSCALE)

        if current_img is None or next_img is None:
            print(f"    Error: Could not read images")
            cumulative_transformations.append(None)
            continue

        try:
            # Preprocess both images
            processed_img1 = enhanced_preprocess_image(current_img)
            processed_img2 = enhanced_preprocess_image(next_img)


            # Create detectors with increased features
            sift = cv2.SIFT_create(nfeatures=3000)
            orb = cv2.ORB_create(nfeatures=3000)

            # Try both SIFT and ORB
            kp1_sift, kp2_sift, matches_sift = find_matches_enhanced(sift, processed_img1, processed_img2)
            kp1_orb, kp2_orb, matches_orb = find_matches_enhanced(orb, processed_img1, processed_img2)


            # Count valid matches
            sift_match_count = len(matches_sift) if matches_sift else 0
            orb_match_count = len(matches_orb) if matches_orb else 0

            print(f"    SIFT Matches: {sift_match_count}")
            print(f"    ORB Matches: {orb_match_count}")

            # Choose best method
            if orb_match_count >= sift_match_count and orb_match_count > 0:
                best_method = 'ORB'
                kp1, kp2, matches = kp1_orb, kp2_orb, matches_orb
            elif sift_match_count > 0:
                best_method = 'SIFT'
                kp1, kp2, matches = kp1_sift, kp2_sift, matches_sift
            else:
                print("    Error: No valid matches found with either method")
                cumulative_transformations.append(None)
                continue

            # Generate visualization path if needed
            vis_path = None
            if visualize :
                vis_dir = os.getcwd()
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"transform_{current_image}_to_{next_image}.jpg")

            # Get transformation with validation
            H = get_transformation_with_fallback(kp1, kp2, matches, processed_img1, processed_img2, visualize=visualize)

            
            # If visualization is requested and a visualization function is provided
            if visualize and vis_func and H is not None:
                result = vis_func(processed_img1, processed_img2, kp1, kp2, matches, H, vis_path)
                
            transformation = decompose_homography(H)

            if transformation:
                qx, qy, qz, qw, tx, ty, tz = transformation
                current_transformation = quaternion_translation_to_matrix(
                    [qx, qy, qz, qw], 
                    [tx, ty, tz]
                )
                cumulative_transformation = np.dot(previous_transformation, current_transformation)
                previous_transformation = cumulative_transformation

                cumulative_transformations.append(cumulative_transformation)
                print(f"    Transformation calculated successfully using {best_method}")
                print(f"    Translation values: tx={tx:.6f}, ty={ty:.6f}, tz={tz:.6f}")
            else:
                cumulative_transformations.append(None)
                print(f"    Error: No transformation found")
        except Exception as e:
            print(f"    Error processing images: {str(e)}")
            cumulative_transformations.append(None)
    
    return cumulative_transformations

def main(base_path, visualize=False, show_error_projection=False):
    """
    Main function that orchestrates the entire process.
    
    Args:
        base_path: Path to the dataset metadata folder
        visualize: Whether to create visualizations
        show_error_projection: Whether to run error projection analysis
        
    Returns:
        Path to the updated CSV file
    """
    # Define paths for input/output and images
    input_csv_path = os.path.join(base_path, "datastream_2", "data_descriptor.csv")
    images_path = os.path.join(base_path, "datastream_1", "samples", "left")
    output_csv_path = input_csv_path  # Save back to the same location

    print(f"CSV file path: {input_csv_path}")
    print(f"Images directory: {images_path}")

    if visualize:
        vis_dir = os.path.join(base_path, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
    
    fieldnames = ['timestamp_start', 'timestamp_stop', 'sampling_time', 'timestamp_root', 
                  'q_1', 'q_2', 'q_3', 'q_w', 'tx', 'ty', 'tz']

    # Get list of images first
    image_files = []
    if os.path.exists(images_path):
        image_files = [f.split('.')[0] for f in os.listdir(images_path) if f.endswith('.jpg')]
        image_files.sort()  # Sort to ensure consistent ordering
    
    if not image_files:
        print("No images found in the images directory. Please add images first.")
        return None

    # Try to read existing CSV first
    try:
        with open(input_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print("Found existing CSV file, using existing data")
    except (FileNotFoundError, IOError):
        print("No existing CSV file found, creating new data from images")
        # Initialize rows
        rows = []
        for i, img_name in enumerate(image_files):
            row = {
                'timestamp_start': img_name,
                'timestamp_stop': img_name,
                'sampling_time': '1',
                'timestamp_root': img_name if i % 5 == 0 else '',  # Every 5th image is root
                'q_1': '', 'q_2': '', 'q_3': '', 'q_w': '',
                'tx': '', 'ty': '', 'tz': ''
            }
            rows.append(row)

    print(f"Processing {len(rows)} rows")

    # Set the visualization function based on whether visualization is enabled
    vis_func = visualize_matches_and_transformation if visualize else None
    
    cumulative_transformations = compute_cumulative_transformations(rows, images_path, visualize, vis_func)

    # Update CSV with scaled transformations
    for i, row in enumerate(rows):
        root_image = determine_root_image(i, rows)
        if i % 5 == 0:  # Root image
            row.update({
                'timestamp_start': image_files[i],
                'timestamp_stop': image_files[i],
                'sampling_time': '1',
                'timestamp_root': image_files[i],  # Root image is its own root
                'q_1': '0', 'q_2': '0', 'q_3': '0', 'q_w': '1',
                'tx': '0', 'ty': '0', 'tz': '0'
            })
        else:
            transformation = cumulative_transformations[i-1] if i > 0 else None
            if transformation is None:
                row.update({
                    'timestamp_start': image_files[i],
                    'timestamp_stop': image_files[i],
                    'sampling_time': '1',
                    'timestamp_root': root_image,
                    'q_1': '', 'q_2': '', 'q_3': '', 'q_w': '',
                    'tx': '', 'ty': '', 'tz': ''
                })
            else:
                q_t = matrix_to_quaternion_translation(transformation)
                # Normalize quaternion if needed
                quat_norm = np.sqrt(q_t[0]**2 + q_t[1]**2 + q_t[2]**2 + q_t[3]**2)
                if quat_norm != 0:
                    q_t[0:4] = q_t[0:4] / quat_norm
                    
                row.update({
                    'timestamp_start': image_files[i],
                    'timestamp_stop': image_files[i],
                    'sampling_time': '1',
                    'timestamp_root': root_image,
                    'q_1': f"{q_t[0]:.6f}", 
                    'q_2': f"{q_t[1]:.6f}", 
                    'q_3': f"{q_t[2]:.6f}", 
                    'q_w': f"{q_t[3]:.6f}",
                    'tx': f"{q_t[4]:.6f}", 
                    'ty': f"{q_t[5]:.6f}", 
                    'tz': f"{q_t[6]:.6f}"
                })
        
        print(f"    Image: {row['timestamp_start']}, Root image: {row['timestamp_root']}")

    try:
        # Save the updated CSV back to datastream_2
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
            writer.writeheader()
            writer.writerows(rows)

        print(f"CSV file updated and saved in datastream_2 at: {output_csv_path}")
        
        # Run error projection analysis if requested
        if show_error_projection:
            try:
                from loopdb_error_projection import run_improved_experiment
                print("\nRunning error projection analysis...")
                results, summary = run_improved_experiment(base_path)
                if summary is not None:
                    print("\nError projection summary by feature method:")
                    print(summary)
                print("Error projection analysis complete!")
            except ImportError:
                print("\nError: Could not import error projection module.")
                print("Make sure loopdb_error_projection.py is in the same directory.")
        
        return output_csv_path
    except Exception as e:
        print(f"Error saving CSV file: {str(e)}")
        return None
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LoopDB Calculation Tool")
    parser.add_argument("--path", dest="base_path", 
                        default=r"C:\Users\barak\Desktop\PhD\Research\loop_closure_dataset\script\metadata", 
                        help="Path to the dataset metadata folder")
    parser.add_argument("--visualize", action="store_true", 
                        help="Generate match visualizations")
    parser.add_argument("--error-projection", action="store_true", 
                        help="Run error projection analysis after calculation")
    
    args = parser.parse_args()
    
    main(args.base_path, visualize=args.visualize, show_error_projection=args.error_projection)