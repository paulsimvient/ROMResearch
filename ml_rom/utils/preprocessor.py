"""Preprocessing utilities for ML-ROM."""
from typing import Dict, Any
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Preprocessor:
    """Preprocessor for ML-ROM data."""
    
    def __init__(self, method: str = "standard"):
        """
        Initialize preprocessor.
        
        Args:
            method: Preprocessing method ("standard", "minmax", "pca", or "none")
        """
        self.method = method
        self.scaler = None
        self.pca = None
        self.fitted = False
    
    def fit(self, data: np.ndarray) -> None:
        """Fit preprocessor on data."""
        if self.method == "standard":
            self.scaler = StandardScaler()
            self.scaler.fit(data)
        elif self.method == "minmax":
            self.scaler = MinMaxScaler()
            self.scaler.fit(data)
        elif self.method == "pca":
            self.scaler = StandardScaler()
            scaled = self.scaler.fit_transform(data)
            self.pca = PCA()
            self.pca.fit(scaled)
        elif self.method == "none":
            pass
        else:
            raise ValueError(f"Unknown preprocessing method: {self.method}")
        
        self.fitted = True
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        if self.method == "standard" or self.method == "minmax":
            return self.scaler.transform(data)
        elif self.method == "pca":
            scaled = self.scaler.transform(data)
            return self.pca.transform(scaled)
        elif self.method == "none":
            return data
        else:
            raise ValueError(f"Unknown preprocessing method: {self.method}")
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted.")
        
        if self.method == "pca":
            data = self.pca.inverse_transform(data)
            return self.scaler.inverse_transform(data)
        elif self.method == "standard" or self.method == "minmax":
            return self.scaler.inverse_transform(data)
        elif self.method == "none":
            return data
        else:
            raise ValueError(f"Unknown preprocessing method: {self.method}")
    
    def get_params(self) -> Dict[str, Any]:
        """Get preprocessor parameters."""
        params = {"method": self.method, "fitted": self.fitted}
        
        if self.scaler is not None:
            if hasattr(self.scaler, "mean_"):
                params["mean"] = self.scaler.mean_.tolist()
            if hasattr(self.scaler, "scale_"):
                params["scale"] = self.scaler.scale_.tolist()
        
        if self.pca is not None:
            params["n_components"] = self.pca.n_components
            params["explained_variance_ratio"] = self.pca.explained_variance_ratio_.tolist()
        
        return params
