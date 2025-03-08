import cv2
import numpy as np
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys


def quaternion_translation_to_homography(q, t, K=None):
    # Convert to numpy arrays if needed
    q = np.array(q)
    t = np.array(t)
    
    # Convert quaternion to rotation matrix
    rot_matrix = R.from_quat(q).as_matrix()
    
    # If no camera matrix provided, estimate one based on image dimensions
    if K is None:
        # for a normal field of view camera
        focal = 1000  # Conservative estimate
        K = np.array([
            [focal, 0, 500],
            [0, focal, 500],
            [0, 0, 1]
        ])
    
    n = np.array([0, 0, 1])  # Normal to plane (assuming frontal plane)
    d = 1.0                  # Distance to plane
    
    # Scale translation based on your scene scale
    scale = 0.01  # Adjust this based on your scene scale
    t = t * scale
    
    # Correct homography formula for planar scene
    H = K @ (rot_matrix - np.outer(t, n)/d) @ np.linalg.inv(K)
    
    return H

def calculate_projection_error(img1, img2, H_computed, H_ground_truth, visualize=False, visualization_func=None):
    # Detect keypoints in first image
    detector = cv2.SIFT_create(nfeatures=500)  # Reduced number of features
    kp1, _ = detector.detectAndCompute(img1, None)
    
    # Select points that are likely to be valid in both views
    h, w = img1.shape[:2]
    margin = int(min(h, w) * 0.1)  # 10% margin to avoid edge effects
    
    # Filter keypoints to avoid edge points that might project outside the image
    filtered_kp = [kp for kp in kp1 if margin < kp.pt[0] < w-margin and margin < kp.pt[1] < h-margin]
    
    # Limit the number of points to avoid overwhelming visualization
    if len(filtered_kp) > 100:
        filtered_kp = filtered_kp[:100]
    
    points = np.float32([kp.pt for kp in filtered_kp]).reshape(-1, 1, 2)
    
    # Project using computed homography
    projected_computed = cv2.perspectiveTransform(points, H_computed)
    
    # Project using ground truth homography
    projected_ground_truth = cv2.perspectiveTransform(points, H_ground_truth)
    
    # Calculate error between the two projections (in pixels)
    errors = np.sqrt(np.sum((projected_computed - projected_ground_truth) ** 2, axis=2))
    
    # Calculate statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)
    
    # If the errors are unreasonably large, something is wrong
    if mean_error > 100:  # More than 100 pixels error is suspicious
        print(f"WARNING: Very large mean error: {mean_error:.2f}px - check transformation calculations")
    
    if visualize and visualization_func:
        # Create visualization showing the projection differences
        visualization = visualization_func(
            img1, img2, points, projected_computed, projected_ground_truth
        )
        cv2.imwrite("projection_error.jpg", visualization)
    
    return {
        "mean_error": mean_error,
        "median_error": median_error, 
        "max_error": max_error,
        "error_distribution": errors.flatten()
    }

def run_projection_error_experiment(csv_path, images_dir, output_dir, visualization_module=None):
    # Import needed functions from visualization module if provided
    create_projection_error_visualization = None
    create_root_visualization = None
    
    if visualization_module:
        if hasattr(visualization_module, 'create_projection_error_visualization'):
            create_projection_error_visualization = visualization_module.create_projection_error_visualization
        if hasattr(visualization_module, 'create_root_visualization'):
            create_root_visualization = visualization_module.create_root_visualization
    
    # Try to import from local module if not provided and needed
    if create_projection_error_visualization is None:
        try:
            from loopdb_visualization import create_projection_error_visualization
        except ImportError:
            print("Warning: create_projection_error_visualization not available")
    
    if create_root_visualization is None:
        try:
            from loopdb_visualization import create_root_visualization
        except ImportError:
            print("Warning: create_root_visualization not available")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV data
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Initialize results storage
    results = {
        'scene_id': [],
        'image_pair': [],
        'feature_method': [],
        'mean_error': [],
        'median_error': [],
        'max_error': [],
        'match_count': []
    }
    
    # Import needed calculation functions
    try:
        from loopdb_calculation import enhanced_preprocess_image, find_matches_enhanced, get_transformation_with_fallback
    except ImportError:
        print("Error: Could not import required functions from loopdb_calculation")
        return None, None
    
    # Process each scene (every 5 images is a scene)
    for scene_start in range(0, len(rows), 5):
        scene_id = scene_start // 5
        scene_rows = rows[scene_start:min(scene_start+5, len(rows))]
        
        if len(scene_rows) < 5:  # Need all 5 images for a complete scene
            print(f"Scene {scene_id+1} has fewer than 5 images, skipping")
            continue
        
        print(f"\nProcessing Scene {scene_id+1} with {len(scene_rows)} images")
        
        # Load all images in the scene first
        scene_images = []
        for i, row in enumerate(scene_rows):
            image_name = row['timestamp_start']
            image_path = os.path.join(images_dir, f"{image_name}.jpg")
            
            if not os.path.exists(image_path):
                print(f"Error: Image {image_name} not found, skipping scene")
                break
                
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error: Could not read image {image_name}, skipping scene")
                break
                
            scene_images.append((image_name, img))
        
        # Skip if we couldn't load all images
        if len(scene_images) != len(scene_rows):
            continue
            
        # Process for each detector type
        for detector_name in ['SIFT', 'ORB']:
            print(f"\nProcessing with {detector_name} detector")
            
            # Create detector
            if detector_name == 'SIFT':
                detector = cv2.SIFT_create(nfeatures=3000)
            else:
                detector = cv2.ORB_create(nfeatures=3000)
            
            # Store visualizations for this detector's sequence
            sequence_visuals = []
            
            # Process all consecutive pairs in the scene
            for i in range(len(scene_images) - 1):
                curr_name, curr_img = scene_images[i]
                next_name, next_img = scene_images[i+1]
                
                print(f"  Processing pair: {curr_name} -> {next_name}")
                
                # Preprocess images
                processed_curr = enhanced_preprocess_image(curr_img)
                processed_next = enhanced_preprocess_image(next_img)
                
                # Get ground truth transformation from metadata
                next_row = scene_rows[i+1]
                if '' in [next_row.get('q_1', ''), next_row.get('tx', '')]:
                    print(f"    Warning: No ground truth transformation, skipping pair")
                    continue
                    
                gt_quat = [
                    float(next_row['q_1']), 
                    float(next_row['q_2']), 
                    float(next_row['q_3']), 
                    float(next_row['q_w'])
                ]
                
                gt_trans = [
                    float(next_row['tx']), 
                    float(next_row['ty']), 
                    float(next_row['tz'])
                ]
                
                # Get camera matrix (estimate if not available)
                h, w = processed_curr.shape
                K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]])
                
                # Convert ground truth to homography
                H_ground_truth = quaternion_translation_to_homography(gt_quat, gt_trans, K)
                
                # Find matches
                kp1, kp2, matches = find_matches_enhanced(detector, processed_curr, processed_next)
                
                if kp1 is None or kp2 is None or len(matches) < 10:
                    print(f"    {detector_name}: Not enough matches, skipping")
                    
                    # Create a blank visualization as placeholder
                    blank_vis = np.zeros((480, 640, 3), dtype=np.uint8)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(blank_vis, f"Pair {i}: {curr_name} -> {next_name}", 
                               (20, 40), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(blank_vis, "Not enough matches to compute transformation", 
                               (20, 80), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    sequence_visuals.append(blank_vis)
                    continue
                
                # Get computed transformation
                H_computed = get_transformation_with_fallback(
                    kp1, kp2, matches, processed_curr, processed_next, visualize=False
                )
                
                if H_computed is None:
                    print(f"    {detector_name}: Failed to compute transformation, skipping")
                    
                    # Create a blank visualization as placeholder
                    blank_vis = np.zeros((480, 640, 3), dtype=np.uint8)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(blank_vis, f"Pair {i}: {curr_name} -> {next_name}", 
                               (20, 40), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(blank_vis, "Failed to compute transformation", 
                               (20, 80), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    sequence_visuals.append(blank_vis)
                    continue
                
                # Calculate projection error
                error_metrics = calculate_projection_error(
                    processed_curr, processed_next, H_computed, H_ground_truth, visualize=False
                )
                
                # Create visualization for this pair
                if create_projection_error_visualization:
                    visualization = create_projection_error_visualization(
                        processed_curr, processed_next,
                        np.float32([kp.pt for kp in kp1]).reshape(-1, 1, 2),
                        cv2.perspectiveTransform(np.float32([kp.pt for kp in kp1]).reshape(-1, 1, 2), H_computed),
                        cv2.perspectiveTransform(np.float32([kp.pt for kp in kp1]).reshape(-1, 1, 2), H_ground_truth)
                    )
                    
                    # Add a title to the visualization
                    title_h = 40
                    title_img = np.zeros((title_h, visualization.shape[1], 3), dtype=np.uint8)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(title_img, f"Pair {i}: {curr_name} -> {next_name}", 
                               (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Combine title and visualization
                    pair_visual = np.vstack((title_img, visualization))
                    
                    
                    
                    # Store for sequence visualization
                    sequence_visuals.append(pair_visual)
                
                # Store results
                results['scene_id'].append(scene_id)
                results['image_pair'].append(f"{curr_name}_{next_name}")
                results['feature_method'].append(detector_name)
                results['mean_error'].append(error_metrics['mean_error'])
                results['median_error'].append(error_metrics['median_error'])
                results['max_error'].append(error_metrics['max_error'])
                results['match_count'].append(len(matches))
                
                print(f"    {detector_name}: Mean Error: {error_metrics['mean_error']:.2f}px, Matches: {len(matches)}")
            
            
            
            # Create sequence visualization for this detector
            if len(sequence_visuals) > 0:
                # Get dimensions and resize all to match width
                widths = [img.shape[1] for img in sequence_visuals]
                max_width = max(widths)
                
                # Create a combined image with a title
                title_img = np.zeros((60, max_width, 3), dtype=np.uint8)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(title_img, f"Scene {scene_id+1} - {detector_name} Detector - Projection Error", 
                           (20, 40), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Start with title image
                combined_img = title_img
                
                # Add each visual in the sequence
                for img in sequence_visuals:
                    # Resize if needed to match max width
                    if img.shape[1] != max_width:
                        h, w = img.shape[:2]
                        new_h = int(h * max_width / w)
                        img_resized = cv2.resize(img, (max_width, new_h))
                    else:
                        img_resized = img
                    
                    # Add to combined image
                    combined_img = np.vstack((combined_img, img_resized))
                
                # Save the combined visualization
                sequence_path = os.path.join(
                    output_dir,
                    f"scene{scene_id+1}_{detector_name}_projection_error.jpg"
                )
                cv2.imwrite(sequence_path, combined_img)
                print(f"Created projection error visualization for {detector_name}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, "projection_error_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Generate summary statistics
    if not results_df.empty:
        summary_by_method = results_df.groupby('feature_method').agg({
            'mean_error': ['mean', 'std', 'min', 'max'],
            'match_count': ['mean', 'std', 'min', 'max']
        })
        
        summary_path = os.path.join(output_dir, "projection_error_summary.csv")
        summary_by_method.to_csv(summary_path)
        
        print(f"\nExperiment complete! Results saved to {output_dir}")
        print("\nSummary by feature method:")
        print(summary_by_method)
        
        # Create visualization of error distribution
        # plt.figure(figsize=(10, 6))
        # for method in results_df['feature_method'].unique():
        #     method_data = results_df[results_df['feature_method'] == method]
        #     plt.hist(method_data['mean_error'], alpha=0.5, label=method, bins=20)
        
        # plt.xlabel('Mean Projection Error (pixels)')
        # plt.ylabel('Frequency')
        # plt.title('Distribution of Projection Errors by Feature Method')
        # plt.legend()
        # plt.grid(True, linestyle='--', alpha=0.7)
        # plt.savefig(os.path.join(output_dir, "error_distribution.png"), dpi=300)
        
        return results_df, summary_by_method
    else:
        print("No valid results were generated.")
        return None, None

def run_improved_experiment(base_path):
    csv_path = os.path.join(base_path, "datastream_2", "data_descriptor.csv")
    images_dir = os.path.join(base_path, "datastream_1", "samples", "left")
    output_dir = os.path.join(base_path, "improved_projection_error_experiment")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Try to import visualization functions
        import loopdb_visualization as vis_module
        
        # Run the projection error experiment
        results, summary = run_projection_error_experiment(csv_path, images_dir, output_dir, vis_module)
        return results, summary
    except ImportError:
        print("Warning: Could not import loopdb_visualization module. Using default visualization.")
        # Run without dedicated visualization module
        results, summary = run_projection_error_experiment(csv_path, images_dir, output_dir)
        return results, summary

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = os.path.join(os.path.expanduser('~'), "loopdb_dataset", "metadata")
        
    print(f"Using base path: {base_path}")
    results, summary = run_improved_experiment(base_path)
    print("Experiment complete!")