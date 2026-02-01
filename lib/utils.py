"""Utility functions for SOM-based model fitting.

This module provides helper functions for data preprocessing, feature extraction,
model fitting functions, and evaluation metrics for use with Kohonen SOMs.
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from scipy import stats


def normalize_data(
    data: np.ndarray,
    method: str = 'standard',
    axis: int = 0
) -> Tuple[np.ndarray, dict]:
    """Normalize data for SOM training.
    
    Args:
        data: Input data of shape (n_samples, n_features).
        method: Normalization method ('standard', 'minmax', or 'robust').
        axis: Axis along which to normalize (0 for column-wise).
    
    Returns:
        Tuple of (normalized_data, normalization_params) where params
        can be used to denormalize or normalize new data.
    """
    params = {'method': method}
    
    if method == 'standard':
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        normalized = (data - mean) / std
        params['mean'] = mean
        params['std'] = std
    
    elif method == 'minmax':
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        normalized = (data - min_val) / range_val
        params['min'] = min_val
        params['max'] = max_val
    
    elif method == 'robust':
        median = np.median(data, axis=axis, keepdims=True)
        q75, q25 = np.percentile(data, [75, 25], axis=axis, keepdims=True)
        iqr = q75 - q25
        iqr = np.where(iqr == 0, 1, iqr)
        normalized = (data - median) / iqr
        params['median'] = median
        params['iqr'] = iqr
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def denormalize_data(data: np.ndarray, params: dict) -> np.ndarray:
    """Denormalize data using stored normalization parameters.
    
    Args:
        data: Normalized data.
        params: Dictionary of normalization parameters from normalize_data().
    
    Returns:
        Denormalized data.
    """
    method = params['method']
    
    if method == 'standard':
        return data * params['std'] + params['mean']
    elif method == 'minmax':
        return data * (params['max'] - params['min']) + params['min']
    elif method == 'robust':
        return data * params['iqr'] + params['median']
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def extract_point_features(points: np.ndarray, reference_point: Optional[np.ndarray] = None) -> np.ndarray:
    """Extract features from point data for SOM training.
    
    Args:
        points: Array of points of shape (n_points, n_dims).
        reference_point: Optional reference point for relative features.
                        If None, uses centroid.
    
    Returns:
        Feature array of shape (n_points, n_features).
    """
    if reference_point is None:
        reference_point = np.mean(points, axis=0)
    
    # Distance from reference
    distances = np.linalg.norm(points - reference_point, axis=1, keepdims=True)
    
    # Combine original coordinates with distance features
    features = np.hstack([points, distances])
    
    return features


def fit_line_2d(points: np.ndarray) -> np.ndarray:
    """Fit a 2D line to points using least squares.
    
    Args:
        points: Array of 2D points of shape (n_points, 2).
    
    Returns:
        Array [a, b, c] representing line equation ax + by + c = 0.
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 points to fit a line")
    
    # Use PCA to find the line direction
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # SVD to find principal direction
    _, _, vh = np.linalg.svd(centered)
    direction = vh[0]  # First principal component
    
    # Convert to line equation ax + by + c = 0
    # Normal to the line is perpendicular to direction
    a = -direction[1]
    b = direction[0]
    c = -(a * centroid[0] + b * centroid[1])
    
    # Normalize
    norm = np.sqrt(a**2 + b**2)
    if norm > 0:
        a, b, c = a/norm, b/norm, c/norm
    
    return np.array([a, b, c])


def fit_plane_3d(points: np.ndarray) -> np.ndarray:
    """Fit a 3D plane to points using least squares.
    
    Args:
        points: Array of 3D points of shape (n_points, 3).
    
    Returns:
        Array [a, b, c, d] representing plane equation ax + by + cz + d = 0.
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a plane")
    
    # Use SVD to find the plane normal
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]  # Last singular vector (smallest singular value)
    
    # Plane equation: normal · (p - centroid) = 0
    # ax + by + cz + d = 0
    a, b, c = normal
    d = -np.dot(normal, centroid)
    
    # Normalize
    norm = np.sqrt(a**2 + b**2 + c**2)
    if norm > 0:
        a, b, c, d = a/norm, b/norm, c/norm, d/norm
    
    return np.array([a, b, c, d])


def fit_polynomial(points: np.ndarray, degree: int = 2) -> np.ndarray:
    """Fit a polynomial to 2D points.
    
    Args:
        points: Array of 2D points of shape (n_points, 2) where
               first column is x and second is y.
        degree: Degree of the polynomial.
    
    Returns:
        Polynomial coefficients (highest degree first).
    """
    if len(points) < degree + 1:
        raise ValueError(f"Need at least {degree + 1} points to fit degree {degree} polynomial")
    
    x = points[:, 0]
    y = points[:, 1]
    
    coeffs = np.polyfit(x, y, degree)
    return coeffs


def compute_residuals(
    points: np.ndarray,
    model_params: np.ndarray,
    model_type: str = 'line2d'
) -> np.ndarray:
    """Compute residuals between points and a fitted model.
    
    Args:
        points: Array of points.
        model_params: Model parameters.
        model_type: Type of model ('line2d', 'plane3d', 'polynomial').
    
    Returns:
        Array of residuals (distances from points to model).
    """
    if model_type == 'line2d':
        a, b, c = model_params
        # Distance from point (x, y) to line ax + by + c = 0
        residuals = np.abs(a * points[:, 0] + b * points[:, 1] + c)
    
    elif model_type == 'plane3d':
        a, b, c, d = model_params
        # Distance from point (x, y, z) to plane ax + by + cz + d = 0
        residuals = np.abs(
            a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
        )
    
    elif model_type == 'polynomial':
        x = points[:, 0]
        y = points[:, 1]
        y_pred = np.polyval(model_params, x)
        residuals = np.abs(y - y_pred)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return residuals


def evaluate_model_fit(
    points: np.ndarray,
    model_params: np.ndarray,
    inlier_mask: np.ndarray,
    model_type: str = 'line2d'
) -> dict:
    """Evaluate the quality of a model fit.
    
    Args:
        points: Array of data points.
        model_params: Fitted model parameters.
        inlier_mask: Boolean mask indicating inliers.
        model_type: Type of model.
    
    Returns:
        Dictionary containing evaluation metrics.
    """
    residuals = compute_residuals(points, model_params, model_type)
    inlier_residuals = residuals[inlier_mask]
    
    metrics = {
        'n_inliers': np.sum(inlier_mask),
        'n_outliers': np.sum(~inlier_mask),
        'inlier_ratio': np.mean(inlier_mask),
        'mean_residual': np.mean(residuals),
        'median_residual': np.median(residuals),
        'mean_inlier_residual': np.mean(inlier_residuals) if len(inlier_residuals) > 0 else np.inf,
        'median_inlier_residual': np.median(inlier_residuals) if len(inlier_residuals) > 0 else np.inf,
        'rmse': np.sqrt(np.mean(residuals**2)),
        'rmse_inliers': np.sqrt(np.mean(inlier_residuals**2)) if len(inlier_residuals) > 0 else np.inf
    }
    
    return metrics


def generate_synthetic_data(
    n_inliers: int = 100,
    n_outliers: int = 20,
    noise_level: float = 0.1,
    model_type: str = 'line2d',
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data for testing SOM-based model fitting.
    
    Args:
        n_inliers: Number of inlier points.
        n_outliers: Number of outlier points.
        noise_level: Standard deviation of Gaussian noise for inliers.
        model_type: Type of model ('line2d', 'plane3d').
        random_seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (data_points, ground_truth_params, true_inlier_mask).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if model_type == 'line2d':
        # Generate line: y = 2x + 1
        true_params = np.array([-2.0, 1.0, -1.0])  # -2x + y - 1 = 0
        
        # Inliers
        x_inliers = np.random.uniform(-5, 5, n_inliers)
        y_inliers = 2 * x_inliers + 1 + np.random.normal(0, noise_level, n_inliers)
        inliers = np.column_stack([x_inliers, y_inliers])
        
        # Outliers
        outliers = np.random.uniform(-5, 5, (n_outliers, 2))
        
    elif model_type == 'plane3d':
        # Generate plane: z = x + 2y + 3
        true_params = np.array([-1.0, -2.0, 1.0, -3.0])  # -x - 2y + z - 3 = 0
        
        # Inliers
        x_inliers = np.random.uniform(-5, 5, n_inliers)
        y_inliers = np.random.uniform(-5, 5, n_inliers)
        z_inliers = x_inliers + 2 * y_inliers + 3 + np.random.normal(0, noise_level, n_inliers)
        inliers = np.column_stack([x_inliers, y_inliers, z_inliers])
        
        # Outliers
        outliers = np.random.uniform(-5, 5, (n_outliers, 3))
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Combine and shuffle
    data = np.vstack([inliers, outliers])
    true_mask = np.hstack([np.ones(n_inliers, dtype=bool), np.zeros(n_outliers, dtype=bool)])
    
    # Shuffle
    indices = np.random.permutation(len(data))
    data = data[indices]
    true_mask = true_mask[indices]
    
    return data, true_params, true_mask


def compare_with_ransac(
    data: np.ndarray,
    som_inliers: np.ndarray,
    true_inliers: Optional[np.ndarray] = None
) -> dict:
    """Compare SOM-based fitting with ground truth or RANSAC results.
    
    Args:
        data: Input data points.
        som_inliers: Boolean mask of SOM-detected inliers.
        true_inliers: Optional ground truth inlier mask.
    
    Returns:
        Dictionary with comparison metrics.
    """
    if true_inliers is None:
        return {
            'som_inlier_count': np.sum(som_inliers),
            'som_inlier_ratio': np.mean(som_inliers)
        }
    
    # Confusion matrix elements
    true_positives = np.sum(som_inliers & true_inliers)
    false_positives = np.sum(som_inliers & ~true_inliers)
    true_negatives = np.sum(~som_inliers & ~true_inliers)
    false_negatives = np.sum(~som_inliers & true_inliers)
    
    # Metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(data)
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy
    }
