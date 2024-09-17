# SIFT-Based Common Keypoints Detection and SVD Analysis

## Overview
This project performs feature detection using SIFT (Scale-Invariant Feature Transform) to identify and match keypoints across a sequence of images. It extracts common keypoints from consecutive image pairs, calculates centroids of these points, and uses Singular Value Decomposition (SVD) for further analysis of the keypoint differences.

## Features
- **SIFT Keypoint Detection**: Finds keypoints and descriptors in images using the SIFT algorithm.
- **Feature Matching**: Matches keypoints between consecutive images using a brute force matcher.
- **Common Keypoints Extraction**: Identifies common keypoints across image sequences.
- **Centroid Calculation**: Calculates centroids for keypoint clusters.
- **SVD Analysis**: Performs Singular Value Decomposition (SVD) on the differences between keypoints and centroids.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
