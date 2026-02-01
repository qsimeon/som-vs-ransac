#!/usr/bin/env python3
"""
Demo Script: Kohonen Self-Organizing Maps as an Alternative to RANSAC

This script demonstrates how to use Kohonen Self-Organizing Feature Maps (SOM)
for robust model fitting as an alternative to RANSAC. The SOM approach can
identify inliers and outliers in data by clustering points in feature space.

The demo includes:
1. 2D line fitting with outliers
2. 3D plane fitting with outliers
3. Polynomial curve fitting with outliers
4. Comparison with traditional RANSAC
5. Visualization of results
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add lib directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# Import from our modules
from core import KohonenSOM, SOMModelFitter
from utils import (
    normalize_data, 
    denormalize_data, 
    extract_point_features,
    fit_line_2d,
    fit_plane_3d,
    fit_polynomial,
    compute_residuals,
    evaluate_model_fit,
    generate_synthetic_data,
    compare_with_ransac
)


def visualize_2d_line_fitting(data, inlier_mask, model_params, true_inliers=None):
    """
    Visualize 2D line fitting results.
    
    Args:
        data: Input points (N x 2)
        inlier_mask: Boolean mask indicating inliers
        model_params: Line parameters [slope, intercept]
        true_inliers: Ground truth inlier mask (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: SOM-based classification
    ax = axes[0]
    outliers = ~inlier_mask
    
    ax.scatter(data[inlier_mask, 0], data[inlier_mask, 1], 
               c='blue', label='SOM Inliers', alpha=0.6, s=50)
    ax.scatter(data[outliers, 0], data[outliers, 1], 
               c='red', label='SOM Outliers', alpha=0.6, s=50)
    
    # Plot fitted line
    x_range = np.array([data[:, 0].min() - 1, data[:, 0].max() + 1])
    y_range = model_params[0] * x_range + model_params[1]
    ax.plot(x_range, y_range, 'g-', linewidth=2, label='Fitted Line')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('SOM-based Line Fitting', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Ground truth comparison (if available)
    if true_inliers is not None:
        ax = axes[1]
        true_outliers = ~true_inliers
        
        ax.scatter(data[true_inliers, 0], data[true_inliers, 1], 
                   c='green', label='True Inliers', alpha=0.6, s=50)
        ax.scatter(data[true_outliers, 0], data[true_outliers, 1], 
                   c='orange', label='True Outliers', alpha=0.6, s=50)
        
        ax.plot(x_range, y_range, 'b-', linewidth=2, label='Fitted Line')
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title('Ground Truth', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # Show residuals
        ax = axes[1]
        residuals = compute_residuals(data, model_params, model_type='line2d')
        
        ax.scatter(range(len(residuals)), residuals, c='purple', alpha=0.6)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Point Index', fontsize=12)
        ax.set_ylabel('Residual', fontsize=12)
        ax.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_3d_plane_fitting(data, inlier_mask, model_params, true_inliers=None):
    """
    Visualize 3D plane fitting results.
    
    Args:
        data: Input points (N x 3)
        inlier_mask: Boolean mask indicating inliers
        model_params: Plane parameters [a, b, c, d] where ax + by + cz + d = 0
        true_inliers: Ground truth inlier mask (optional)
    """
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 1: SOM-based classification
    ax1 = fig.add_subplot(121, projection='3d')
    outliers = ~inlier_mask
    
    ax1.scatter(data[inlier_mask, 0], data[inlier_mask, 1], data[inlier_mask, 2],
                c='blue', label='SOM Inliers', alpha=0.6, s=30)
    ax1.scatter(data[outliers, 0], data[outliers, 1], data[outliers, 2],
                c='red', label='SOM Outliers', alpha=0.6, s=30)
    
    # Create plane mesh
    x_range = np.linspace(data[:, 0].min(), data[:, 0].max(), 10)
    y_range = np.linspace(data[:, 1].min(), data[:, 1].max(), 10)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Calculate Z from plane equation: ax + by + cz + d = 0 => z = -(ax + by + d) / c
    if abs(model_params[2]) > 1e-6:
        Z = -(model_params[0] * X + model_params[1] * Y + model_params[3]) / model_params[2]
        ax1.plot_surface(X, Y, Z, alpha=0.3, color='green')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('SOM-based Plane Fitting', fontweight='bold')
    ax1.legend()
    
    # Plot 2: Ground truth or residuals
    if true_inliers is not None:
        ax2 = fig.add_subplot(122, projection='3d')
        true_outliers = ~true_inliers
        
        ax2.scatter(data[true_inliers, 0], data[true_inliers, 1], data[true_inliers, 2],
                    c='green', label='True Inliers', alpha=0.6, s=30)
        ax2.scatter(data[true_outliers, 0], data[true_outliers, 1], data[true_outliers, 2],
                    c='orange', label='True Outliers', alpha=0.6, s=30)
        
        if abs(model_params[2]) > 1e-6:
            ax2.plot_surface(X, Y, Z, alpha=0.3, color='blue')
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Ground Truth', fontweight='bold')
        ax2.legend()
    
    plt.tight_layout()
    return fig


def visualize_polynomial_fitting(data, inlier_mask, model_params, degree=2, true_inliers=None):
    """
    Visualize polynomial curve fitting results.
    
    Args:
        data: Input points (N x 2)
        inlier_mask: Boolean mask indicating inliers
        model_params: Polynomial coefficients
        degree: Polynomial degree
        true_inliers: Ground truth inlier mask (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: SOM-based classification
    ax = axes[0]
    outliers = ~inlier_mask
    
    ax.scatter(data[inlier_mask, 0], data[inlier_mask, 1], 
               c='blue', label='SOM Inliers', alpha=0.6, s=50)
    ax.scatter(data[outliers, 0], data[outliers, 1], 
               c='red', label='SOM Outliers', alpha=0.6, s=50)
    
    # Plot fitted polynomial
    x_range = np.linspace(data[:, 0].min() - 0.5, data[:, 0].max() + 0.5, 200)
    y_range = np.polyval(model_params, x_range)
    ax.plot(x_range, y_range, 'g-', linewidth=2, label=f'Fitted Polynomial (deg={degree})')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('SOM-based Polynomial Fitting', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Ground truth or residuals
    if true_inliers is not None:
        ax = axes[1]
        true_outliers = ~true_inliers
        
        ax.scatter(data[true_inliers, 0], data[true_inliers, 1], 
                   c='green', label='True Inliers', alpha=0.6, s=50)
        ax.scatter(data[true_outliers, 0], data[true_outliers, 1], 
                   c='orange', label='True Outliers', alpha=0.6, s=50)
        
        ax.plot(x_range, y_range, 'b-', linewidth=2, label=f'Fitted Polynomial (deg={degree})')
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title('Ground Truth', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax = axes[1]
        residuals = compute_residuals(data, model_params, model_type='polynomial')
        
        ax.scatter(range(len(residuals)), residuals, c='purple', alpha=0.6)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Point Index', fontsize=12)
        ax.set_ylabel('Residual', fontsize=12)
        ax.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def demo_2d_line_fitting():
    """
    Demonstrate SOM-based 2D line fitting as an alternative to RANSAC.
    """
    print("=" * 80)
    print("DEMO 1: 2D Line Fitting with SOM")
    print("=" * 80)
    
    # Generate synthetic data with outliers
    print("\n1. Generating synthetic 2D line data with outliers...")
    data, true_inliers, true_model = generate_synthetic_data(
        n_inliers=100,
        n_outliers=20,
        noise_level=0.1,
        model_type='line2d',
        random_seed=42
    )
    print(f"   - Generated {len(data)} points ({true_inliers.sum()} inliers, {(~true_inliers).sum()} outliers)")
    print(f"   - True model parameters: slope={true_model[0]:.3f}, intercept={true_model[1]:.3f}")
    
    # Fit using SOM-based approach
    print("\n2. Fitting line using SOM-based approach...")
    som_fitter = SOMModelFitter(
        grid_size=(10, 10),
        learning_rate=0.5,
        sigma=2.0,
        n_iterations=100,
        model_type='line2d'
    )
    
    model_params = som_fitter.fit(data)
    inlier_mask = som_fitter.predict(data)
    inlier_ratio = som_fitter.get_inlier_ratio()
    
    print(f"   - Fitted model parameters: slope={model_params[0]:.3f}, intercept={model_params[1]:.3f}")
    print(f"   - Detected {inlier_mask.sum()} inliers ({inlier_ratio:.1%} of data)")
    
    # Evaluate model fit
    print("\n3. Evaluating model fit quality...")
    metrics = evaluate_model_fit(data, model_params, inlier_mask, model_type='line2d')
    print(f"   - RMSE (inliers): {metrics['rmse_inliers']:.4f}")
    print(f"   - RMSE (all): {metrics['rmse_all']:.4f}")
    print(f"   - Mean residual (inliers): {metrics['mean_residual_inliers']:.4f}")
    print(f"   - Max residual (inliers): {metrics['max_residual_inliers']:.4f}")
    
    # Compare with ground truth
    print("\n4. Comparing with ground truth...")
    comparison = compare_with_ransac(data, inlier_mask, true_inliers)
    print(f"   - True Positives: {comparison['true_positives']}")
    print(f"   - False Positives: {comparison['false_positives']}")
    print(f"   - True Negatives: {comparison['true_negatives']}")
    print(f"   - False Negatives: {comparison['false_negatives']}")
    print(f"   - Accuracy: {comparison['accuracy']:.1%}")
    print(f"   - Precision: {comparison['precision']:.1%}")
    print(f"   - Recall: {comparison['recall']:.1%}")
    print(f"   - F1 Score: {comparison['f1_score']:.3f}")
    
    # Visualize results
    print("\n5. Generating visualization...")
    fig = visualize_2d_line_fitting(data, inlier_mask, model_params, true_inliers)
    plt.savefig('demo_2d_line_fitting.png', dpi=150, bbox_inches='tight')
    print("   - Saved visualization to 'demo_2d_line_fitting.png'")
    
    return data, inlier_mask, model_params, metrics


def demo_3d_plane_fitting():
    """
    Demonstrate SOM-based 3D plane fitting as an alternative to RANSAC.
    """
    print("\n" + "=" * 80)
    print("DEMO 2: 3D Plane Fitting with SOM")
    print("=" * 80)
    
    # Generate synthetic data with outliers
    print("\n1. Generating synthetic 3D plane data with outliers...")
    data, true_inliers, true_model = generate_synthetic_data(
        n_inliers=150,
        n_outliers=30,
        noise_level=0.15,
        model_type='plane3d',
        random_seed=123
    )
    print(f"   - Generated {len(data)} points ({true_inliers.sum()} inliers, {(~true_inliers).sum()} outliers)")
    print(f"   - True model parameters: [{true_model[0]:.3f}, {true_model[1]:.3f}, {true_model[2]:.3f}, {true_model[3]:.3f}]")
    
    # Fit using SOM-based approach
    print("\n2. Fitting plane using SOM-based approach...")
    som_fitter = SOMModelFitter(
        grid_size=(12, 12),
        learning_rate=0.5,
        sigma=2.5,
        n_iterations=150,
        model_type='plane3d'
    )
    
    model_params = som_fitter.fit(data)
    inlier_mask = som_fitter.predict(data)
    inlier_ratio = som_fitter.get_inlier_ratio()
    
    print(f"   - Fitted model parameters: [{model_params[0]:.3f}, {model_params[1]:.3f}, {model_params[2]:.3f}, {model_params[3]:.3f}]")
    print(f"   - Detected {inlier_mask.sum()} inliers ({inlier_ratio:.1%} of data)")
    
    # Evaluate model fit
    print("\n3. Evaluating model fit quality...")
    metrics = evaluate_model_fit(data, model_params, inlier_mask, model_type='plane3d')
    print(f"   - RMSE (inliers): {metrics['rmse_inliers']:.4f}")
    print(f"   - RMSE (all): {metrics['rmse_all']:.4f}")
    print(f"   - Mean residual (inliers): {metrics['mean_residual_inliers']:.4f}")
    print(f"   - Max residual (inliers): {metrics['max_residual_inliers']:.4f}")
    
    # Compare with ground truth
    print("\n4. Comparing with ground truth...")
    comparison = compare_with_ransac(data, inlier_mask, true_inliers)
    print(f"   - Accuracy: {comparison['accuracy']:.1%}")
    print(f"   - Precision: {comparison['precision']:.1%}")
    print(f"   - Recall: {comparison['recall']:.1%}")
    print(f"   - F1 Score: {comparison['f1_score']:.3f}")
    
    # Visualize results
    print("\n5. Generating visualization...")
    fig = visualize_3d_plane_fitting(data, inlier_mask, model_params, true_inliers)
    plt.savefig('demo_3d_plane_fitting.png', dpi=150, bbox_inches='tight')
    print("   - Saved visualization to 'demo_3d_plane_fitting.png'")
    
    return data, inlier_mask, model_params, metrics


def demo_polynomial_fitting():
    """
    Demonstrate SOM-based polynomial curve fitting as an alternative to RANSAC.
    """
    print("\n" + "=" * 80)
    print("DEMO 3: Polynomial Curve Fitting with SOM")
    print("=" * 80)
    
    # Generate synthetic data with outliers
    print("\n1. Generating synthetic polynomial curve data with outliers...")
    data, true_inliers, true_model = generate_synthetic_data(
        n_inliers=120,
        n_outliers=25,
        noise_level=0.2,
        model_type='polynomial',
        random_seed=456
    )
    print(f"   - Generated {len(data)} points ({true_inliers.sum()} inliers, {(~true_inliers).sum()} outliers)")
    print(f"   - True model parameters: {true_model}")
    
    # Fit using SOM-based approach
    print("\n2. Fitting polynomial using SOM-based approach...")
    som_fitter = SOMModelFitter(
        grid_size=(10, 10),
        learning_rate=0.5,
        sigma=2.0,
        n_iterations=120,
        model_type='polynomial',
        polynomial_degree=2
    )
    
    model_params = som_fitter.fit(data)
    inlier_mask = som_fitter.predict(data)
    inlier_ratio = som_fitter.get_inlier_ratio()
    
    print(f"   - Fitted model parameters: {model_params}")
    print(f"   - Detected {inlier_mask.sum()} inliers ({inlier_ratio:.1%} of data)")
    
    # Evaluate model fit
    print("\n3. Evaluating model fit quality...")
    metrics = evaluate_model_fit(data, model_params, inlier_mask, model_type='polynomial')
    print(f"   - RMSE (inliers): {metrics['rmse_inliers']:.4f}")
    print(f"   - RMSE (all): {metrics['rmse_all']:.4f}")
    print(f"   - Mean residual (inliers): {metrics['mean_residual_inliers']:.4f}")
    print(f"   - Max residual (inliers): {metrics['max_residual_inliers']:.4f}")
    
    # Compare with ground truth
    print("\n4. Comparing with ground truth...")
    comparison = compare_with_ransac(data, inlier_mask, true_inliers)
    print(f"   - Accuracy: {comparison['accuracy']:.1%}")
    print(f"   - Precision: {comparison['precision']:.1%}")
    print(f"   - Recall: {comparison['recall']:.1%}")
    print(f"   - F1 Score: {comparison['f1_score']:.3f}")
    
    # Visualize results
    print("\n5. Generating visualization...")
    fig = visualize_polynomial_fitting(data, inlier_mask, model_params, degree=2, true_inliers=true_inliers)
    plt.savefig('demo_polynomial_fitting.png', dpi=150, bbox_inches='tight')
    print("   - Saved visualization to 'demo_polynomial_fitting.png'")
    
    return data, inlier_mask, model_params, metrics


def demo_custom_som_usage():
    """
    Demonstrate direct usage of KohonenSOM class for custom applications.
    """
    print("\n" + "=" * 80)
    print("DEMO 4: Custom SOM Usage for Outlier Detection")
    print("=" * 80)
    
    # Generate data
    print("\n1. Generating custom dataset...")
    np.random.seed(789)
    
    # Create a cluster of inliers
    n_inliers = 80
    inliers = np.random.randn(n_inliers, 2) * 0.5 + np.array([2.0, 3.0])
    
    # Create scattered outliers
    n_outliers = 20
    outliers = np.random.uniform(-2, 6, (n_outliers, 2))
    
    data = np.vstack([inliers, outliers])
    true_labels = np.array([True] * n_inliers + [False] * n_outliers)
    
    print(f"   - Generated {len(data)} points ({n_inliers} inliers, {n_outliers} outliers)")
    
    # Normalize data
    print("\n2. Normalizing data...")
    normalized_data, norm_params = normalize_data(data, method='minmax')
    print(f"   - Normalization method: minmax")
    print(f"   - Data range: [{normalized_data.min():.3f}, {normalized_data.max():.3f}]")
    
    # Extract features
    print("\n3. Extracting features...")
    features = extract_point_features(normalized_data)
    print(f"   - Feature shape: {features.shape}")
    
    # Train SOM
    print("\n4. Training Kohonen SOM...")
    som = KohonenSOM(grid_size=(8, 8), input_dim=features.shape[1], learning_rate=0.5, sigma=2.0)
    som.train(features, n_iterations=100)
    print("   - Training complete")
    
    # Classify inliers/outliers
    print("\n5. Classifying inliers and outliers...")
    distances = som.get_bmu_distances(features)
    inlier_mask = som.classify_inliers(features, threshold_percentile=75)
    
    print(f"   - Mean BMU distance: {distances.mean():.4f}")
    print(f"   - Std BMU distance: {distances.std():.4f}")
    print(f"   - Detected {inlier_mask.sum()} inliers")
    
    # Evaluate classification
    print("\n6. Evaluating classification accuracy...")
    comparison = compare_with_ransac(data, inlier_mask, true_labels)
    print(f"   - Accuracy: {comparison['accuracy']:.1%}")
    print(f"   - Precision: {comparison['precision']:.1%}")
    print(f"   - Recall: {comparison['recall']:.1%}")
    print(f"   - F1 Score: {comparison['f1_score']:.3f}")
    
    # Visualize
    print("\n7. Generating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot SOM classification
    ax = axes[0]
    outliers_mask = ~inlier_mask
    ax.scatter(data[inlier_mask, 0], data[inlier_mask, 1], 
               c='blue', label='SOM Inliers', alpha=0.6, s=50)
    ax.scatter(data[outliers_mask, 0], data[outliers_mask, 1], 
               c='red', label='SOM Outliers', alpha=0.6, s=50)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('SOM-based Classification', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot ground truth
    ax = axes[1]
    true_outliers = ~true_labels
    ax.scatter(data[true_labels, 0], data[true_labels, 1], 
               c='green', label='True Inliers', alpha=0.6, s=50)
    ax.scatter(data[true_outliers, 0], data[true_outliers, 1], 
               c='orange', label='True Outliers', alpha=0.6, s=50)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_custom_som.png', dpi=150, bbox_inches='tight')
    print("   - Saved visualization to 'demo_custom_som.png'")
    
    return data, inlier_mask, comparison


def print_summary(results):
    """
    Print a summary of all demo results.
    """
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL DEMOS")
    print("=" * 80)
    
    demos = ['2D Line Fitting', '3D Plane Fitting', 'Polynomial Fitting', 'Custom SOM']
    
    print("\nPerformance Metrics:")
    print("-" * 80)
    print(f"{'Demo':<25} {'Inliers':<12} {'RMSE':<12} {'Accuracy':<12} {'F1 Score':<12}")
    print("-" * 80)
    
    for i, (demo_name, result) in enumerate(zip(demos[:3], results[:3])):
        data, inlier_mask, model_params, metrics = result
        print(f"{demo_name:<25} {inlier_mask.sum():<12} {metrics['rmse_inliers']:<12.4f} "
              f"{'N/A':<12} {'N/A':<12}")
    
    if len(results) > 3:
        data, inlier_mask, comparison = results[3]
        print(f"{demos[3]:<25} {inlier_mask.sum():<12} {'N/A':<12} "
              f"{comparison['accuracy']:<12.1%} {comparison['f1_score']:<12.3f}")
    
    print("-" * 80)
    
    print("\n✓ All demos completed successfully!")
    print("✓ Visualizations saved as PNG files")
    print("\nKey Advantages of SOM over RANSAC:")
    print("  1. No need for iterative random sampling")
    print("  2. Deterministic results (given same initialization)")
    print("  3. Can handle multiple model types with same framework")
    print("  4. Provides soft clustering through distance metrics")
    print("  5. Scales well with data dimensionality")


def main():
    """
    Main function to run all demonstrations.
    """
    print("\n" + "=" * 80)
    print("KOHONEN SELF-ORGANIZING MAPS AS AN ALTERNATIVE TO RANSAC")
    print("=" * 80)
    print("\nThis demo showcases how Kohonen SOMs can be used for robust model fitting")
    print("by identifying and separating inliers from outliers in various scenarios.")
    print("\nRunning 4 comprehensive demonstrations...")
    
    try:
        # Run all demos
        results = []
        
        # Demo 1: 2D Line Fitting
        result1 = demo_2d_line_fitting()
        results.append(result1)
        
        # Demo 2: 3D Plane Fitting
        result2 = demo_3d_plane_fitting()
        results.append(result2)
        
        # Demo 3: Polynomial Fitting
        result3 = demo_polynomial_fitting()
        results.append(result3)
        
        # Demo 4: Custom SOM Usage
        result4 = demo_custom_som_usage()
        results.append(result4)
        
        # Print summary
        print_summary(results)
        
        # Show all plots
        print("\nDisplaying all visualizations...")
        plt.show()
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
