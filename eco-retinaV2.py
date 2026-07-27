import pandas as pd
import numpy as np 
import psutil
import concurrent.futures
import statsmodels.api as sm
import warnings

from typing import List, Optional, Tuple, Dict, Union
from numpy.typing import NDArray
from itertools import permutations
from codecarbon import EmissionsTracker

warnings.filterwarnings("ignore")

# Custom utility imports
from utils import (
    find_combinations, precompute_powers, generate_features, 
    get_feature_hashes, compute_abs_correlations, get_reg_sets, 
    filter_reg_sets, remove_elements_from_arrays, concurrent_regressions, 
    sort_array, get_intervals
)

class EcoRETINA:
    """
    EcoRETINA: An innovative, eco-friendly algorithm specifically designed for out-of-sample prediction. 
    It functions as a regression-based flexible approximator, linear in parameters but nonlinear in inputs, 
    utilizing a selective model search to optimize performance.
    """

    def __init__(self):
        """Initialize EcoRETINA with placeholders for model attributes."""
        self.model_indices: Optional[NDArray] = None
        self.params: List[float] = []
        self.chunk_size: int = 500
        self.sm_model = None
        self.best_score: float = float('inf')
        self.combinations: List = []
        self.X_total: Optional[NDArray] = None
        self.X: Optional[NDArray] = None
        self.y: Optional[NDArray] = None
        
        # Attributs pour la gestion des zéros
        self.handle_zeros: str = 'prevent_division'
        self.epsilon: Union[float, str] = 'auto'  # 'auto' applique la règle du min + 1
        self.con_cols_indices: List[int] = []
        self.cols_with_zeros_: List[int] = []
        self.translations_: Dict[int, float] = {}

    def fit(self, 
            y: NDArray, 
            X: NDArray, 
            con_cols_indices: List[int], 
            dummy_cols_indices: List[int], 
            col_names: Optional[List[str]] = None, 
            params: List[float] = [-1.0, 0.0, 1.0], 
            cross_dummy: bool = False, 
            max_r2: float = 0.9, 
            grid: float = 0.005, 
            reg_type: str = 'linear', 
            loss: str = 'mse',
            max_instances: int = 100000, 
            max_reg: int = 100, 
            model_step: int = 1, 
            chunk_size: int = 500, 
            seed: int = 8, 
            cov_type: str = 'nonrobust',
            handle_zeros: str = 'prevent_division', 
            epsilon: Union[float, str] = 'auto') -> None:
        """
        Fit the EcoRETINA model on a dataset using a grid-based subset selection strategy.
        """
        
        tracker = EmissionsTracker(tracking_mode='process', output_file='eco_retina_emissions.csv', project_name='Eco-RETINA')
        tracker.start()
        
        self.params = params
        self.chunk_size = chunk_size
        self.handle_zeros = handle_zeros
        self.epsilon = epsilon
        self.con_cols_indices = con_cols_indices
        
        # 1. Data Preparation and Subsampling
        X, y = self._prepare_data(X, y, max_instances, seed)

        # 2. Handle Zeros (Translation globale ou Identification pour prévention)
        X = self._handle_zeros_in_data(X)

        # 3. Feature Generation (Polynomials & Interactions)
        combinations_list = find_combinations(con_cols_indices=self.con_cols_indices, dummy_cols_indices=dummy_cols_indices, params=self.params, cross_dummy=cross_dummy)
        
        # Filtrage des divisions par zéro
        if self.handle_zeros == 'prevent_division' and self.cols_with_zeros_:
            filtered_combos = []
            for combo in combinations_list:
                if len(combo) == 4:
                    a, b, c, d = combo
                    if (a in self.cols_with_zeros_ and c < 0) or (b in self.cols_with_zeros_ and d < 0):
                        continue
                filtered_combos.append(combo)
            combinations_list = filtered_combos

        with np.errstate(divide='ignore', invalid='ignore'):
            precomputed_powers = precompute_powers(X, self.params)
            
        X_total = generate_features(X, combinations_list, precomputed_powers, self.params, self.chunk_size)
        hashes = get_feature_hashes(X_total)

        # 4. Format Feature Names
        variables_df = self._generate_feature_names(col_names, combinations_list, hashes, X_total.shape[1])

        # 5. Split data into 3 chunks and compute correlations
        X_chunks, y_chunks, corr_indices_dic, X_chunks_sorted_dic = self._split_and_correlate(X_total, y, hashes)

        # 6. Multithreaded Regression Set Extraction
        reg_set_list_filt, indices_list = self._extract_and_filter_reg_sets(X_chunks, X_chunks_sorted_dic, corr_indices_dic, max_r2, grid, max_reg)

        # 7. Evaluate Permutations (Subsample Cross-Validation)
        model_indices = self._evaluate_subsamples(X_chunks, y_chunks, reg_set_list_filt, corr_indices_dic, loss, model_step)

        # 8. Finalize Best Model & Fit Statsmodels
        self._fit_final_model(X_total, y, variables_df, model_indices, reg_type, cov_type)
        
        tracker.stop()

    def _prepare_data(self, X: NDArray, y: NDArray, max_instances: int, seed: int) -> Tuple[NDArray, NDArray]:
        """Shuffle dataset, subset to max_instances, and conditionally drop rows with zeros."""
        n_rows = X.shape[0]
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_rows)

        y_sub = np.take(y, indices, axis=0)[:max_instances]
        X_sub = np.take(X, indices, axis=0)[:max_instances]

        if self.handle_zeros == 'drop_rows':
            mask = (X_sub[:, self.con_cols_indices] != 0).all(axis=1)
            return X_sub[mask], y_sub[mask]
        
        return X_sub, y_sub

    def _handle_zeros_in_data(self, X: NDArray) -> NDArray:
        """Applique la translation globale ou marque les colonnes à risque."""
        self.cols_with_zeros_ = []
        self.translations_ = {}
        
        for col in self.con_cols_indices:
            if (X[:, col] == 0).any():
                if self.handle_zeros == 'translate':
                    
                    if self.epsilon == 'auto':
                        # Translation intelligente : on s'assure que le nouveau min sera 1
                        shift = np.abs(X[:, col].min()) + 1.0
                    else:
                        # Comportement de secours manuel
                        shift = float(self.epsilon)
                        while (X[:, col] + shift == 0).any():
                            shift += float(self.epsilon)
                    
                    X[:, col] += shift
                    self.translations_[col] = shift
                    
                elif self.handle_zeros == 'prevent_division':
                    self.cols_with_zeros_.append(col)
                    
        return X

    def predict(self, X: NDArray, confidence: float = 0.95) -> NDArray:
        """Predict using the trained EcoRETINA model and compute prediction intervals."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        # 1. Appliquer rigoureusement les mêmes translations apprises à l'entraînement
        if self.handle_zeros == 'translate':
            for col, shift in self.translations_.items():
                X[:, col] += shift

        # 2. Sécurisation : gérer les zéros "surprises" qui n'existaient pas à l'entraînement
        for col in self.con_cols_indices:
            if (X[:, col] == 0).any():
                if self.handle_zeros == 'translate' and col not in self.translations_:
                    # Si c'est un zéro complètement nouveau sur cette colonne
                    if self.epsilon == 'auto':
                        shift = np.abs(X[:, col].min()) + 1.0
                    else:
                        shift = float(self.epsilon)
                        while (X[:, col] + shift == 0).any():
                            shift += float(self.epsilon)
                    
                    warnings.warn(f"Zéro inattendu dans la colonne {col} à l'inférence. Application d'une translation globale de {shift}.")
                    X[:, col] += shift
                    
                elif self.handle_zeros == 'prevent_division' and col not in self.cols_with_zeros_:
                    # Substitution locale uniquement pour éviter que l'inférence ne crashe (puisqu'on ne peut pas filtrer les combinaisons après coup)
                    fallback_val = 1e-5 if self.epsilon == 'auto' else float(self.epsilon)
                    warnings.warn(f"Zéro inattendu dans la colonne {col} à l'inférence. Remplacement local par {fallback_val} pour éviter le crash.")
                    X[:, col] = np.where(X[:, col] == 0, fallback_val, X[:, col])
                    
        with np.errstate(divide='ignore', invalid='ignore'):
            precomputed_powers = precompute_powers(X, self.params)
            
        X_transformed = generate_features(X, self.combinations, precomputed_powers, self.params, self.chunk_size)
        
        y_pred = self.sm_model.predict(X_transformed)
        
        self.pi_lower, self.pi_upper, self.ci_lower, self.ci_upper = get_intervals(
            y_train=self.y, 
            X_train=self.X, 
            beta=self.sm_model.params.values, 
            X_new=X_transformed, 
            confidence=confidence
        )

        return y_pred

    # ---------------------------------------------------------
    # Les autres méthodes (_generate_feature_names, _split_and_correlate, 
    # _extract_and_filter_reg_sets, _evaluate_subsamples, 
    # _fit_final_model, load_emissions_report) restent identiques
    # --------------------------------------------------------- 