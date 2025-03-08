import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import csv
from scipy.spatial.transform import Rotation as R

def visualize_matches_and_transformation(img1, img2, kp1, kp2, matches, H, save_path=None):
    """Visualize feature matches between two images with a clean, high-resolution output."""
    try:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if len(img1.shape) == 2:
            img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        else:
            img1_color = img1.copy()
            
        if len(img2.shape) == 2:
            img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        else:
            img2_color = img2.copy()
        
        match_img_height = max(h1, h2)
        match_img_width = w1 + w2
        match_img = np.zeros((match_img_height, match_img_width, 3), dtype=np.uint8)
        
        match_img[0:h1, 0:w1] = img1_color
        match_img[0:h2, w1:w1+w2] = img2_color
        
        match_count = min(50, len(matches))
        selected_matches = matches[:match_count]
        np.random.seed(42)
        
        for i, m in enumerate(selected_matches):
            pt1 = tuple(map(int, kp1[m.queryIdx].pt))
            pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
            
            color = tuple(map(int, np.random.randint(100, 255, 3).tolist()))
            
            cv2.circle(match_img, pt1, 5, color, -1)
            cv2.circle(match_img, pt2, 5, color, -1)
            
            cv2.line(match_img, pt1, pt2, color, 2)
        
        target_width = 2400
        h, w = match_img.shape[:2]
        
        if w != target_width:
            scale_factor = target_width / w
            target_height = int(h * scale_factor)
            match_img = cv2.resize(match_img, (target_width, target_height), 
                                  interpolation=cv2.INTER_LANCZOS4)
        
        h_pad = 40
        titled_img = np.zeros((match_img.shape[0] + h_pad, match_img.shape[1], 3), dtype=np.uint8)
        titled_img[h_pad:, :] = match_img
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(titled_img, f"Feature Matches: {len(matches)} matches found", 
                   (10, 30), font, 1.0, (255, 255, 255), 2)
        
        if save_path:
            if not save_path.lower().endswith('.png'):
                save_path = save_path.rsplit('.', 1)[0] + '.png'
            
            cv2.imwrite(save_path, titled_img)
            print(f"    Visualization saved to {save_path}")
        
        return titled_img
    
    except Exception as e:
        print(f"    Visualization error: {str(e)}")
        return None

def create_projection_error_visualization(img1, img2, points, projected_computed, projected_ground_truth):
    """Visualize the difference between computed and ground truth projections."""
    result_img = cv2.cvtColor(img2.copy(), cv2.COLOR_GRAY2BGR)
    
    errors = np.sqrt(np.sum((projected_computed - projected_ground_truth) ** 2, axis=2)).flatten()
    
    std_dev = np.std(errors)
    mean_err = np.mean(errors)
    valid_idx = errors < (mean_err + 3 * std_dev)
    
    if np.sum(valid_idx) > 0:
        filtered_errors = errors[valid_idx]
        max_error = np.max(filtered_errors)
    else:
        max_error = np.max(errors) if len(errors) > 0 else 1.0
    
    cmap = plt.cm.jet
    
    for i in range(len(points)):
        error_val = errors[i]
        
        if error_val > (mean_err + 3 * std_dev):
            continue
            
        error_ratio = min(error_val / max_error, 1.0) if max_error > 0 else 0
        
        color = tuple(int(255 * c) for c in cmap(error_ratio)[:3][::-1])
        
        x_comp, y_comp = projected_computed[i][0]
        x_gt, y_gt = projected_ground_truth[i][0]
        
        pt_computed = (int(x_comp), int(y_comp))
        pt_ground_truth = (int(x_gt), int(y_gt))
        
        h, w = result_img.shape[:2]
        
        if (0 <= pt_computed[0] < w and 0 <= pt_computed[1] < h and
            0 <= pt_ground_truth[0] < w and 0 <= pt_ground_truth[1] < h):
            
            cv2.circle(result_img, pt_computed, 4, color, -1)
            cv2.circle(result_img, pt_ground_truth, 4, (0, 255, 0), 1)
            
            if error_val < 100:
                cv2.line(result_img, pt_computed, pt_ground_truth, color, 2)
    
    h, w = result_img.shape[:2]
    colorbar_width = 30
    colorbar = np.zeros((h, colorbar_width, 3), dtype=np.uint8)
    
    for i in range(h):
        pos = 1.0 - (i / h)
        color_bgr = [int(255 * c) for c in cmap(pos)[:3][::-1]]
        colorbar[i, :] = color_bgr
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    cv2.putText(colorbar, f"Max: {max_error:.1f}px", (2, 20), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(colorbar, f"Min: 0px", (2, h-10), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    result_with_colorbar = np.hstack((result_img, colorbar))
    
    title_bar = np.zeros((40, result_with_colorbar.shape[1], 3), dtype=np.uint8)
    filtered_mean = np.mean(filtered_errors) if np.sum(valid_idx) > 0 else mean_err
    cv2.putText(title_bar, f"Projection Error Visualization (Mean: {filtered_mean:.2f}px)", 
                (10, 30), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
    final_visualization = np.vstack((title_bar, result_with_colorbar))
    
    return final_visualization

def create_root_visualization(root_name, root_img, scene_id, reference_width):
    """Create a visualization for the root image matching the style of projection error visualizations."""
    processed_root = enhanced_preprocess_image(root_img)
    
    h, w = processed_root.shape
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    colored_img = cv2.cvtColor(processed_root, cv2.COLOR_GRAY2BGR)
    
    if w != reference_width - 30:
        new_w = reference_width - 30
        new_h = int(h * (new_w / w))
        colored_img = cv2.resize(colored_img, (new_w, new_h))
    
    np.random.seed(42)
    
    h, w = colored_img.shape[:2]
    step_x = max(1, w // 20)
    step_y = max(1, h // 20)
    
    for y in range(step_y, h - step_y, step_y):
        for x in range(step_x, w - step_x, step_x):
            x_jitter = int(np.random.normal(0, step_x/3))
            y_jitter = int(np.random.normal(0, step_y/3))
            
            pt_x = min(max(0, x + x_jitter), w-1)
            pt_y = min(max(0, y + y_jitter), h-1)
            
            g = 200 + np.random.randint(0, 56)
            r = np.random.randint(0, 100)
            b = np.random.randint(0, 100)
            
            cv2.circle(colored_img, (pt_x, pt_y), 4, (b, g, r), -1)
    
    colorbar_width = 30
    colorbar = np.zeros((colored_img.shape[0], colorbar_width, 3), dtype=np.uint8)
    
    cmap = plt.cm.jet
    for i in range(colorbar.shape[0]):
        pos = 1.0 - (i / colorbar.shape[0])
        color = cmap(pos)
        bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
        colorbar[i, :] = bgr
    
    cv2.putText(colorbar, "Max", (2, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(colorbar, "Min", (2, colorbar.shape[0]-10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    result = np.hstack([colored_img, colorbar])
    
    error_text_img = np.zeros((40, result.shape[1], 3), dtype=np.uint8)
    cv2.putText(error_text_img, f"Projection Error Visualization (Mean: 0.00px)", 
               (10, 25), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    result = np.vstack([error_text_img, result])
    
    title_h = 40
    title_img = np.zeros((title_h, result.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_img, f"Pair R: {root_name} (Root)", 
               (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    final_vis = np.vstack([title_img, result])
    
    return final_vis

def visualize_all_images_from_csv(csv_path, images_dir, output_path):
    """Create an overview visualization of all scenes from a CSV file."""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    scenes = {}
    for row in rows:
        timestamp = row['timestamp_start']
        root = row['timestamp_root']
        
        if not root:
            continue
            
        if root not in scenes:
            scenes[root] = []
            
        scenes[root].append(timestamp)
    
    sorted_scenes = sorted(scenes.items())
    
    max_images_per_row = 5
    padding = 10
    max_width = 0
    max_height = 0
    
    for root, images in sorted_scenes:
        for img_name in images:
            img_path = os.path.join(images_dir, f"{img_name}.jpg")
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            max_width = max(max_width, w)
            max_height = max(max_height, h)
    
    thumb_width = 300
    thumb_height = int(thumb_width * max_height / max_width)
    
    num_scenes = len(sorted_scenes)
    grid_width = min(max_images_per_row, max([len(images) for _, images in sorted_scenes]))
    grid_height = sum([min(max_images_per_row, len(images)) for _, _ in sorted_scenes])
    
    canvas_width = grid_width * (thumb_width + padding) + padding
    canvas_height = num_scenes * (thumb_height + padding * 3) + padding
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    y_offset = padding
    for root, images in sorted_scenes:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, f"Scene: {root}", (padding, y_offset + 20), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        y_offset += padding * 2
        
        x_offset = padding
        for i, img_name in enumerate(images):
            if i >= max_images_per_row:
                break
                
            img_path = os.path.join(images_dir, f"{img_name}.jpg")
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            thumb = cv2.resize(img, (thumb_width, thumb_height))
            
            y1 = y_offset
            y2 = y_offset + thumb_height
            x1 = x_offset
            x2 = x_offset + thumb_width
            
            if y2 < canvas_height and x2 < canvas_width:
                canvas[y1:y2, x1:x2] = thumb
                cv2.putText(canvas, img_name, (x1, y2 + 15), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                x_offset += thumb_width + padding
        
        y_offset += thumb_height + padding * 2
    
    cv2.imwrite(output_path, canvas)
    print(f"Visualization saved to {output_path}")

def enhanced_preprocess_image(img):
    """Advanced preprocessing to improve feature matching."""
    max_dim = 1200
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    
    img_norm = cv2.normalize(img_clahe, None, 0, 255, cv2.NORM_MINMAX)
    
    img_blur = cv2.GaussianBlur(img_norm, (3, 3), 0)
    
    img_edges = cv2.Laplacian(img_blur, cv2.CV_8U, ksize=3)
    img_enhanced = cv2.addWeighted(img_blur, 0.7, img_edges, 0.3, 0)
    
    return img_enhanced

def preprocess_image(img):
    """Basic preprocessing with CLAHE and normalization."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    
    img_norm = cv2.normalize(img_clahe, None, 0, 255, cv2.NORM_MINMAX)
    
    return img_norm

def quaternion_translation_to_homography(q, t, K=None):
    """Convert quaternion and translation to a homography matrix for planar scenes."""
    q = np.array(q)
    t = np.array(t)
    
    rot_matrix = R.from_quat(q).as_matrix()
    
    if K is None:
        focal = 1000
        K = np.array([
            [focal, 0, 500],
            [0, focal, 500],
            [0, 0, 1]
        ])
    
    n = np.array([0, 0, 1])
    d = 1.0
    
    scale = 0.01
    t = t * scale
    
    H = K @ (rot_matrix - np.outer(t, n)/d) @ np.linalg.inv(K)
    
    return H

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LoopDB Visualization Tool")
    parser.add_argument("--csv", required=True, help="Path to metadata CSV file")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--output", default="./visualization_output.jpg", help="Path to save visualization (default: ./visualization_output.jpg)")
    parser.add_argument("--img1", help="First image name for match visualization")
    parser.add_argument("--img2", help="Second image name for match visualization")
    parser.add_argument("--detector", choices=["SIFT", "ORB"], default="SIFT", help="Feature detector to use (default: SIFT)")
    
    args = parser.parse_args()
    
    if args.img1 and args.img2:
        print(f"Visualizing matches between {args.img1} and {args.img2} using {args.detector}...")
        
        img1_path = os.path.join(args.images, f"{args.img1}.jpg")
        img2_path = os.path.join(args.images, f"{args.img2}.jpg")
        
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            print(f"Error: One or both images not found")
            sys.exit(1)
            
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        img1 = enhanced_preprocess_image(img1)
        img2 = enhanced_preprocess_image(img2)
        
        if args.detector == 'SIFT':
            detector = cv2.SIFT_create(nfeatures=3000)
        else:
            detector = cv2.ORB_create(nfeatures=3000)
            
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)
        
        if args.detector == 'SIFT':
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            good_matches = bf.match(des1, des2)
            good_matches = sorted(good_matches, key=lambda x: x.distance)[:100]
            
        if len(good_matches) < 10:
            print(f"Warning: Only {len(good_matches)} good matches found, visualization may be poor")
            
        if len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            output_path = os.path.join(os.getcwd(), f"{args.img1}_to_{args.img2}_matches.jpg")
            visualize_matches_and_transformation(img1, img2, kp1, kp2, good_matches, H, output_path)
            print(f"Match visualization saved to: {output_path}")
        else:
            print("Error: Not enough matches to compute homography")
    else:
        output_path = os.path.join(os.getcwd(), os.path.basename(args.output))
        print(f"Generating dataset overview visualization...")
        visualize_all_images_from_csv(args.csv, args.images, output_path)
        print(f"Dataset overview saved to: {output_path}")