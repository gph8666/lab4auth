#!/usr/bin/env python3
"""
Hybrid Fingerprint Matching System with SVM
Combines features from Minutiae, Orientation Field, and Ridge Count Matching.
Trains on training data and evaluates on testing data using SVM to minimize EER.
Calculates EER, FAR, FRR, and their min, max, and average values.
"""

import os
import cv2
import pandas as pd
import numpy as np
from skimage.morphology import skeletonize
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Image enhancement
def enhance_image(image_path):
    """Enhance contrast and apply adaptive thresholding."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    equ = cv2.equalizeHist(img)
    return cv2.adaptiveThreshold(equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 181, 11)

# Minutiae Matching
def calculate_minutiae_similarity(image_a, image_b):
    img_a = enhance_image(image_a)
    img_b = enhance_image(image_b)

    skeleton_a = skeletonize(img_a // 255)
    skeleton_b = skeletonize(img_b // 255)

    coords_a = np.argwhere(skeleton_a)
    coords_b = np.argwhere(skeleton_b)

    centroid_a = np.mean(coords_a, axis=0)
    centroid_b = np.mean(coords_b, axis=0)

    distances_a = np.linalg.norm(coords_a - centroid_a, axis=1)
    distances_b = np.linalg.norm(coords_b - centroid_b, axis=1)

    min_size = min(len(distances_a), len(distances_b))
    distances_a = distances_a[:min_size]
    distances_b = distances_b[:min_size]

    diff = np.abs(distances_a - distances_b)
    return np.sum(diff < 7) / len(diff)

# Orientation Field Matching
def calculate_orientation_similarity(image_a, image_b):
    img_a = enhance_image(image_a)
    img_b = enhance_image(image_b)

    grad_a = cv2.Sobel(img_a, cv2.CV_64F, 1, 0, ksize=3) + cv2.Sobel(img_a, cv2.CV_64F, 0, 1, ksize=3)
    grad_b = cv2.Sobel(img_b, cv2.CV_64F, 1, 0, ksize=3) + cv2.Sobel(img_b, cv2.CV_64F, 0, 1, ksize=3)

    grad_a = grad_a.flatten()
    grad_b = grad_b.flatten()
    min_size = min(len(grad_a), len(grad_b))
    grad_a = grad_a[:min_size]
    grad_b = grad_b[:min_size]

    return np.corrcoef(grad_a, grad_b)[0, 1]

# Ridge Count Matching
def calculate_ridge_similarity(image_a, image_b):
    img_a = enhance_image(image_a)
    img_b = enhance_image(image_b)

    count_a = np.sum(img_a // 255)
    count_b = np.sum(img_b // 255)

    if max(count_a, count_b) == 0:
        return 0

    return 1 - abs(count_a - count_b) / max(count_a, count_b)

# Extract features for SVM
def extract_features(pairs):
    """Extract features for each pair using all three methods."""
    features = []
    for file_a, file_b in pairs:
        f1 = calculate_minutiae_similarity(file_a, file_b)
        f2 = calculate_orientation_similarity(file_a, file_b)
        f3 = calculate_ridge_similarity(file_a, file_b)
        features.append([f1, f2, f3])  # Combine features from all methods
    return np.array(features)

# Calculate EER, FAR, and FRR
def calculate_metrics(y_true, y_scores):
    """Calculate EER, FAR, and FRR with their thresholds."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    eer_threshold = interp1d(fpr, thresholds)(eer)
    far = fpr  # False Acceptance Rate
    frr = 1 - tpr  # False Rejection Rate
    return eer, eer_threshold, far, frr

# Main execution
if __name__ == "__main__":
    train_dir = "/home/student/Downloads/nist/sd04/png_txt/figs_0"  # Directory containing training fingerprint PNG files
    test_dir = "/home/student/Downloads/nist/sd04/png_txt/figs_6"   # Directory containing testing fingerprint PNG files
    output_file = "metrics_hybrid_svm.csv"

    # List and pair training files dynamically
    train_files_f = sorted([f for f in os.listdir(train_dir) if f.startswith('f') and f.endswith('.png')])
    train_files_s = sorted([f for f in os.listdir(train_dir) if f.startswith('s') and f.endswith('.png')])

    # Create genuine pairs for training
    train_genuine_pairs = [(os.path.join(train_dir, f), os.path.join(train_dir, s)) 
                           for f, s in zip(train_files_f, train_files_s)]

    # Generate impostor pairs for training (circular mismatch)
    train_impostor_pairs = []
    for i, file_f in enumerate(train_files_f):
        mismatched_index = (i + 1) % len(train_files_s)  # Simple circular mismatch
        train_impostor_pairs.append((os.path.join(train_dir, file_f), os.path.join(train_dir, train_files_s[mismatched_index])))

    # Combine training pairs
    train_pairs = train_genuine_pairs + train_impostor_pairs
    train_labels = np.concatenate([np.ones(len(train_genuine_pairs)), np.zeros(len(train_impostor_pairs))])

    # Extract training features
    train_features = extract_features(train_pairs)

    # List and pair testing files dynamically
    test_files_f = sorted([f for f in os.listdir(test_dir) if f.startswith('f') and f.endswith('.png')])
    test_files_s = sorted([f for f in os.listdir(test_dir) if f.startswith('s') and f.endswith('.png')])

    # Create genuine pairs for testing
    test_genuine_pairs = [(os.path.join(test_dir, f), os.path.join(test_dir, s)) 
                          for f, s in zip(test_files_f, test_files_s)]

    # Generate impostor pairs for testing (circular mismatch)
    test_impostor_pairs = []
    for i, file_f in enumerate(test_files_f):
        mismatched_index = (i + 1) % len(test_files_s)  # Simple circular mismatch
        test_impostor_pairs.append((os.path.join(test_dir, file_f), os.path.join(test_dir, test_files_s[mismatched_index])))

    # Combine testing pairs
    test_pairs = test_genuine_pairs + test_impostor_pairs
    test_labels = np.concatenate([np.ones(len(test_genuine_pairs)), np.zeros(len(test_impostor_pairs))])

    # Extract testing features
    test_features = extract_features(test_pairs)

    # Train SVM model
    clf = SVC(probability=True, kernel="linear")
    clf.fit(train_features, train_labels)

    # Predict on test set
    test_scores = clf.predict_proba(test_features)[:, 1]

    # Calculate metrics
    eer, eer_threshold, far, frr = calculate_metrics(test_labels, test_scores)

    # Calculate min, max, and avg for FAR and FRR
    far_min = np.min(far)
    far_max = np.max(far)
    far_avg = np.mean(far)
    frr_min = np.min(frr)
    frr_max = np.max(frr)
    frr_avg = np.mean(frr)

    # Evaluate performance
    accuracy = accuracy_score(test_labels, (test_scores >= eer_threshold).astype(int))

    # Save metrics
    metrics_df = pd.DataFrame({
        "Metric": ["EER", "Threshold", "Accuracy", "FAR Min", "FAR Max", "FAR Avg", "FRR Min", "FRR Max", "FRR Avg"],
        "Value": [eer, eer_threshold, accuracy, far_min, far_max, far_avg, frr_min, frr_max, frr_avg]
    })
    metrics_df.to_csv(output_file, index=False)

    # Print metrics
    print("Hybrid Fingerprint Matching with SVM Metrics:")
    print(metrics_df)

