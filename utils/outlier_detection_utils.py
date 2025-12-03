import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import RANSACRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import base64

def detect_outliers_series(
    df,
    op_state_col,
    feature,
    method='isoforest',
    contamination=0.01,
    n_neighbors=20,
    iqr_factor=1.5,
    percentile_cut=0.01,
    residual_threshold=3.5
):
    """
    Calculates outlier flags for a single feature against an operational state.
    Returns a boolean Series (True = Outlier).
    """
    # Prepare Data
    X = df[[op_state_col, feature]].copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna()
    
    if X.empty:
        return pd.Series(False, index=df.index)

    idx = X.index
    outlier_labels = np.ones(len(X)) # 1 = Inlier, -1 = Outlier

    if method == 'isoforest':
        model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
        outlier_labels = model.fit_predict(X)
        
    elif method == 'lof':
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1)
        outlier_labels = lof.fit_predict(X)
        
    elif method == 'iqr':
        vals = X[feature]
        Q1 = vals.quantile(0.25)
        Q3 = vals.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_factor * IQR
        upper = Q3 + iqr_factor * IQR
        outlier_labels = np.where((vals < lower) | (vals > upper), -1, 1)
        
    elif method == 'percentile':
        vals = X[feature]
        lower = vals.quantile(percentile_cut)
        upper = vals.quantile(1 - percentile_cut)
        outlier_labels = np.where((vals < lower) | (vals > upper), -1, 1)

    elif method == 'residual':
        # RANSAC / Residual Logic
        x_val = X[[op_state_col]].values
        y_val = X[feature].values
        
        if len(x_val) > 50: # RANSAC needs enough samples
            try:
                ransac = RANSACRegressor(random_state=42)
                ransac.fit(x_val, y_val)
                predictions = ransac.predict(x_val)
                residuals = np.abs(y_val - predictions)
                
                # Robust Z-Score (MAD)
                median_residual = np.median(residuals)
                mad = np.median(np.abs(residuals - median_residual))
                if mad == 0: mad = 1e-6
                
                modified_z_scores = 0.6745 * (residuals - median_residual) / mad
                outlier_labels = np.where(modified_z_scores > residual_threshold, -1, 1)
            except:
                outlier_labels = np.ones(len(X)) # Fallback if RANSAC fails
        else:
            outlier_labels = np.ones(len(X))

    # Return boolean mask aligned to original index
    is_outlier = pd.Series(False, index=df.index)
    is_outlier.loc[idx] = (outlier_labels == -1)
    return is_outlier

def detect_multivariate_outliers(df, feature_cols, method='pca_recon', contamination=0.01):
    """
    Detects outliers using the entire feature set at once.
    Returns a boolean Series (True = Outlier).
    """
    X = df[feature_cols].dropna()
    if X.empty:
        return pd.Series(False, index=df.index)
    
    # Scaling is mandatory for multivariate distance methods
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    outlier_labels = np.ones(len(X))

    if method == 'pca_recon':
        # PCA Reconstruction Error
        n_components = min(len(feature_cols), 10) # Cap components
        if n_components < 1: n_components = 1
        
        # Keep 95% variance or max components
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)
        X_recon = pca.inverse_transform(X_pca)
        
        # Reconstruction Error (Euclidean distance)
        recon_error = np.sqrt(np.sum((X_scaled - X_recon) ** 2, axis=1))
        
        # Threshold
        threshold = np.quantile(recon_error, 1 - contamination)
        outlier_labels = np.where(recon_error > threshold, -1, 1)
        
    elif method == 'isoforest_global':
        iso = IsolationForest(n_estimators=300, contamination=contamination, n_jobs=-1, random_state=42)
        outlier_labels = iso.fit_predict(X_scaled)
        
    # Align results
    is_outlier = pd.Series(False, index=df.index)
    is_outlier.loc[X.index] = (outlier_labels == -1)
    return is_outlier

def generate_outlier_plots(df_original, mask_outliers, strategy, op_state=None, plot_cols=None):
    """
    Generates a list of (Title, Base64_Image) tuples for reports/display.
    df_original: DataFrame with data
    mask_outliers: Boolean Series (True = Row is Outlier)
    """
    plot_images = []
    
    # Downsample for plotting speed
    MAX_POINTS = 5000
    if len(df_original) > MAX_POINTS:
        sample_idx = np.random.choice(df_original.index, MAX_POINTS, replace=False)
        # Ensure we include some outliers in the sample if they exist
        outlier_indices = df_original[mask_outliers].index
        if len(outlier_indices) > 0:
            # Mix random sample with outliers (up to a limit)
            outlier_sample = np.random.choice(outlier_indices, min(len(outlier_indices), 1000), replace=False)
            final_idx = np.unique(np.concatenate([sample_idx, outlier_sample]))
            plot_df = df_original.loc[final_idx].copy()
            plot_mask = mask_outliers.loc[final_idx]
        else:
            plot_df = df_original.loc[sample_idx].copy()
            plot_mask = mask_outliers.loc[sample_idx]
    else:
        plot_df = df_original.copy()
        plot_mask = mask_outliers

    inliers = plot_df[~plot_mask]
    outliers = plot_df[plot_mask]

    # --- Strategy 1: Pairwise Visuals ---
    if strategy == 'pairwise' and op_state and plot_cols:
        for feat in plot_cols:
            if feat not in plot_df.columns: continue
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Plot Inliers
            sns.scatterplot(
                x=inliers[op_state], y=inliers[feat], 
                color='gray', alpha=0.3, s=15, label='Inlier', ax=ax, edgecolor=None
            )
            # Plot Outliers
            if not outliers.empty:
                sns.scatterplot(
                    x=outliers[op_state], y=outliers[feat], 
                    color='red', alpha=0.8, s=25, label='Outlier', ax=ax, marker='x'
                )
            
            ax.set_title(f"Outlier Detection: {op_state} vs {feat}")
            ax.legend()
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            
            # Save
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plot_images.append((f"{feat} vs {op_state}", img_str))
            plt.close(fig)

    # --- Strategy 2: Multivariate Visuals ---
    elif strategy == 'multivariate' and plot_cols:
        # 1. PCA Projection (2D)
        try:
            # We calculate PCA on the plotting subset
            scaler = StandardScaler()
            subset_vals = scaler.fit_transform(plot_df[plot_cols].dropna())
            
            pca = PCA(n_components=2)
            pca_res = pca.fit_transform(subset_vals)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Color by outlier status
            colors = np.where(plot_mask.loc[plot_df.dropna().index], 'red', 'gray')
            alphas = np.where(plot_mask.loc[plot_df.dropna().index], 0.8, 0.3)
            
            ax.scatter(pca_res[:, 0], pca_res[:, 1], c=colors, alpha=0.5, s=20, edgecolor=None)
            
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Multivariate Outliers (PCA Projection)")
            ax.grid(True, alpha=0.2)
            
            # Save
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plot_images.append(("PCA Projection 2D", img_str))
            plt.close(fig)
            
        except Exception as e:
            print(f"PCA Plot Error: {e}")

    return plot_images