"""Core module for Kohonen Self-Organizing Maps (SOM) as an alternative to RANSAC.

This module provides the main SOM implementation for robust model fitting,
offering an alternative approach to RANSAC for outlier rejection and
parameter estimation in computer vision and data fitting tasks.
"""

import numpy as np
from typing import Tuple, Optional, Callable, Union
import warnings


class KohonenSOM:
    """Kohonen Self-Organizing Map for robust model fitting.
    
    This class implements a SOM that can be used as an alternative to RANSAC
    for tasks like line fitting, plane fitting, homography estimation, etc.
    The SOM learns to organize data points and identify inliers vs outliers.
    
    Attributes:
        grid_size (Tuple[int, int]): Size of the SOM grid (rows, cols).
        input_dim (int): Dimensionality of input data.
        weights (np.ndarray): Weight vectors for each neuron in the grid.
        learning_rate (float): Initial learning rate for training.
        sigma (float): Initial neighborhood radius.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        input_dim: int = 2,
        learning_rate: float = 0.5,
        sigma: float = None,
        random_seed: Optional[int] = None
    ):
        """Initialize the Kohonen SOM.
        
        Args:
            grid_size: Tuple of (rows, cols) for the SOM grid.
            input_dim: Dimensionality of the input feature vectors.
            learning_rate: Initial learning rate (typically 0.1 to 0.5).
            sigma: Initial neighborhood radius. If None, set to max(grid_size)/2.
            random_seed: Random seed for reproducibility.
        """
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.learning_rate_init = learning_rate
        self.learning_rate = learning_rate
        
        if sigma is None:
            self.sigma_init = max(grid_size) / 2.0
        else:
            self.sigma_init = sigma
        self.sigma = self.sigma_init
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize weights randomly
        self.weights = np.random.randn(grid_size[0], grid_size[1], input_dim)
        
        # Create grid coordinates for distance calculations
        self.grid_coords = self._create_grid_coordinates()
    
    def _create_grid_coordinates(self) -> np.ndarray:
        """Create coordinate matrix for the SOM grid.
        
        Returns:
            Array of shape (rows, cols, 2) containing grid coordinates.
        """
        rows, cols = self.grid_size
        coords = np.zeros((rows, cols, 2))
        for i in range(rows):
            for j in range(cols):
                coords[i, j] = [i, j]
        return coords
    
    def _find_bmu(self, input_vector: np.ndarray) -> Tuple[int, int]:
        """Find the Best Matching Unit (BMU) for an input vector.
        
        Args:
            input_vector: Input feature vector of shape (input_dim,).
        
        Returns:
            Tuple of (row, col) indices of the BMU.
        """
        # Calculate Euclidean distance to all neurons
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx
    
    def _calculate_neighborhood(self, bmu_idx: Tuple[int, int], sigma: float) -> np.ndarray:
        """Calculate neighborhood influence for all neurons.
        
        Args:
            bmu_idx: Index of the Best Matching Unit.
            sigma: Current neighborhood radius.
        
        Returns:
            Array of shape (rows, cols) with neighborhood influence values.
        """
        bmu_coord = np.array([bmu_idx[0], bmu_idx[1]])
        distances = np.linalg.norm(self.grid_coords - bmu_coord, axis=2)
        influence = np.exp(-(distances ** 2) / (2 * sigma ** 2))
        return influence
    
    def train(
        self,
        data: np.ndarray,
        num_iterations: int = 1000,
        decay_function: str = 'exponential'
    ) -> None:
        """Train the SOM on input data.
        
        Args:
            data: Training data of shape (n_samples, input_dim).
            num_iterations: Number of training iterations.
            decay_function: Type of decay ('exponential' or 'linear').
        """
        n_samples = data.shape[0]
        
        for iteration in range(num_iterations):
            # Decay learning rate and neighborhood radius
            if decay_function == 'exponential':
                self.learning_rate = self.learning_rate_init * np.exp(-iteration / num_iterations)
                self.sigma = self.sigma_init * np.exp(-iteration / num_iterations)
            else:  # linear
                self.learning_rate = self.learning_rate_init * (1 - iteration / num_iterations)
                self.sigma = self.sigma_init * (1 - iteration / num_iterations)
            
            # Select random sample
            sample_idx = np.random.randint(0, n_samples)
            input_vector = data[sample_idx]
            
            # Find BMU
            bmu_idx = self._find_bmu(input_vector)
            
            # Calculate neighborhood influence
            influence = self._calculate_neighborhood(bmu_idx, self.sigma)
            
            # Update weights
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    self.weights[i, j] += (
                        self.learning_rate * influence[i, j] * 
                        (input_vector - self.weights[i, j])
                    )
    
    def get_bmu_distances(self, data: np.ndarray) -> np.ndarray:
        """Get distances from each data point to its BMU.
        
        Args:
            data: Input data of shape (n_samples, input_dim).
        
        Returns:
            Array of distances of shape (n_samples,).
        """
        distances = np.zeros(data.shape[0])
        for i, sample in enumerate(data):
            bmu_idx = self._find_bmu(sample)
            distances[i] = np.linalg.norm(sample - self.weights[bmu_idx])
        return distances
    
    def classify_inliers(
        self,
        data: np.ndarray,
        threshold: Optional[float] = None,
        percentile: float = 75.0
    ) -> np.ndarray:
        """Classify data points as inliers or outliers.
        
        Args:
            data: Input data of shape (n_samples, input_dim).
            threshold: Distance threshold for inlier classification.
                      If None, use percentile-based threshold.
            percentile: Percentile for automatic threshold (default 75).
        
        Returns:
            Boolean array of shape (n_samples,) where True indicates inlier.
        """
        distances = self.get_bmu_distances(data)
        
        if threshold is None:
            threshold = np.percentile(distances, percentile)
        
        return distances <= threshold


class SOMModelFitter:
    """SOM-based model fitter as an alternative to RANSAC.
    
    This class uses a Kohonen SOM to robustly fit models to data in the
    presence of outliers, providing an alternative to RANSAC.
    """
    
    def __init__(
        self,
        model_func: Callable,
        feature_extractor: Optional[Callable] = None,
        grid_size: Tuple[int, int] = (10, 10),
        som_iterations: int = 1000,
        inlier_percentile: float = 75.0
    ):
        """Initialize the SOM model fitter.
        
        Args:
            model_func: Function that fits a model to data points.
                       Should accept data and return model parameters.
            feature_extractor: Optional function to extract features from raw data.
                              If None, uses data as-is.
            grid_size: Size of the SOM grid.
            som_iterations: Number of SOM training iterations.
            inlier_percentile: Percentile for inlier threshold determination.
        """
        self.model_func = model_func
        self.feature_extractor = feature_extractor
        self.grid_size = grid_size
        self.som_iterations = som_iterations
        self.inlier_percentile = inlier_percentile
        self.som = None
        self.model_params = None
        self.inlier_mask = None
    
    def fit(
        self,
        data: np.ndarray,
        learning_rate: float = 0.5,
        sigma: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit a model to data using SOM for outlier rejection.
        
        Args:
            data: Input data of shape (n_samples, n_features).
            learning_rate: SOM learning rate.
            sigma: SOM neighborhood radius.
        
        Returns:
            Tuple of (model_parameters, inlier_mask).
        """
        # Extract features if needed
        if self.feature_extractor is not None:
            features = self.feature_extractor(data)
        else:
            features = data
        
        # Determine input dimension
        input_dim = features.shape[1]
        
        # Initialize and train SOM
        self.som = KohonenSOM(
            grid_size=self.grid_size,
            input_dim=input_dim,
            learning_rate=learning_rate,
            sigma=sigma
        )
        self.som.train(features, num_iterations=self.som_iterations)
        
        # Classify inliers
        self.inlier_mask = self.som.classify_inliers(
            features,
            percentile=self.inlier_percentile
        )
        
        # Fit model to inliers
        inlier_data = data[self.inlier_mask]
        
        if len(inlier_data) < 2:
            warnings.warn("Too few inliers detected. Using all data.")
            inlier_data = data
            self.inlier_mask = np.ones(len(data), dtype=bool)
        
        self.model_params = self.model_func(inlier_data)
        
        return self.model_params, self.inlier_mask
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict using the fitted model.
        
        Args:
            data: Input data for prediction.
        
        Returns:
            Model predictions.
        """
        if self.model_params is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # This is a placeholder - actual prediction depends on model type
        return self.model_params
    
    def get_inlier_ratio(self) -> float:
        """Get the ratio of inliers to total data points.
        
        Returns:
            Inlier ratio as a float between 0 and 1.
        """
        if self.inlier_mask is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return np.sum(self.inlier_mask) / len(self.inlier_mask)
