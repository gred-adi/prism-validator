import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import RANSACRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
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

def generate_outlier_plots(df_original, mask_outliers, strategy, op_state=None, plot_cols=None, time_col='DATETIME'):
    """
    Generates a list of dictionaries containing plot details.
    
    Returns structure:
    [
        {
            'type': 'pairwise' | 'multivariate',
            'title': str,
            'scatter_img': base64_str (for pairwise),
            'ts_img': base64_str (for pairwise),
            'img': base64_str (for multivariate, e.g. PCA)
        },
        ...
    ]
    """
    plots_data = []
    
    # Downsample for plotting speed
    MAX_POINTS = 5000
    if len(df_original) > MAX_POINTS:
        sample_idx = np.random.choice(df_original.index, MAX_POINTS, replace=False)
        # Ensure we include some outliers in the sample if they exist
        outlier_indices = df_original[mask_outliers].index
        if len(outlier_indices) > 0:
            # Mix random sample with outliers (up to a limit to avoid overcrowding)
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

    # Ensure DATETIME is properly typed for plotting
    if time_col in plot_df.columns:
        plot_df[time_col] = pd.to_datetime(plot_df[time_col])
        plot_df = plot_df.sort_values(time_col)

    inliers = plot_df[~plot_mask]
    outliers = plot_df[plot_mask]

    # --- Strategy 1: Pairwise Visuals ---
    if strategy == 'pairwise' and op_state and plot_cols:
        for feat in plot_cols:
            if feat not in plot_df.columns: continue
            
            # --- Plot A: Scatter with KDE ---
            fig_scat, ax_scat = plt.subplots(figsize=(8, 5))
            
            # KDE Density Estimation (on Inliers only for clean contour)
            try:
                # Remove NaNs for KDE calculation
                kde_data = inliers[[op_state, feat]].dropna()
                if len(kde_data) > 5:
                    x = kde_data[op_state].values
                    y = kde_data[feat].values
                    
                    # Calculate the point density
                    k = gaussian_kde(np.vstack([x, y]))
                    xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j, y.min():y.max():y.size**0.5*1j]
                    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
                    
                    # Overlay contours
                    ax_scat.contour(xi, yi, zi.reshape(xi.shape), levels=5, colors='black', alpha=0.3, linewidths=0.5)
                    ax_scat.contourf(xi, yi, zi.reshape(xi.shape), levels=5, cmap="Greys", alpha=0.1)
            except Exception as e:
                print(f"KDE Plot Error for {feat}: {e}")

            # Plot Inliers
            sns.scatterplot(
                x=inliers[op_state], y=inliers[feat], 
                color='gray', alpha=0.4, s=15, label='Inlier', ax=ax_scat, edgecolor=None
            )
            # Plot Outliers
            if not outliers.empty:
                sns.scatterplot(
                    x=outliers[op_state], y=outliers[feat], 
                    color='red', alpha=0.9, s=30, label='Outlier', ax=ax_scat, marker='x', linewidth=1.5
                )
            
            ax_scat.set_title(f"Scatter: {op_state} vs {feat}")
            ax_scat.legend()
            ax_scat.grid(True, alpha=0.2)
            plt.tight_layout()
            
            # Save Scatter
            buf_scat = BytesIO()
            fig_scat.savefig(buf_scat, format="png", dpi=100)
            buf_scat.seek(0)
            img_scat = base64.b64encode(buf_scat.read()).decode('utf-8')
            plt.close(fig_scat)

            # --- Plot B: Time Series (Dual Axis) ---
            fig_ts, ax_ts = plt.subplots(figsize=(10, 4))
            
            # Primary Y: Op State (Green)
            ax_ts.plot(plot_df[time_col], plot_df[op_state], color='green', alpha=0.3, label=op_state, linewidth=1)
            ax_ts.set_ylabel(op_state, color='green')
            ax_ts.tick_params(axis='y', labelcolor='green')
            
            # Secondary Y: Feature (Blue)
            ax2 = ax_ts.twinx()
            ax2.plot(plot_df[time_col], plot_df[feat], color='blue', alpha=0.5, label=feat, linewidth=1)
            
            # Highlight Outliers on Secondary Y
            if not outliers.empty:
                ax2.scatter(outliers[time_col], outliers[feat], color='red', s=20, label='Outlier', zorder=10, marker='x')
            
            ax2.set_ylabel(feat, color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            
            ax_ts.set_title(f"Time Series: {feat} & {op_state}")
            ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            # Combined Legend - Bottom Outside
            lines, labels = ax_ts.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            # bbox_to_anchor controls position relative to plot
            # (0.5, -0.2) means centered horizontally (0.5) and below the plot area (-0.2)
            ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
            
            plt.tight_layout()
            
            # Save Time Series
            buf_ts = BytesIO()
            fig_ts.savefig(buf_ts, format="png", dpi=100)
            buf_ts.seek(0)
            img_ts = base64.b64encode(buf_ts.read()).decode('utf-8')
            plt.close(fig_ts)

            # Append structured data
            plots_data.append({
                'type': 'pairwise',
                'title': feat,
                'scatter_img': img_scat,
                'ts_img': img_ts
            })

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
            
            # Save PCA Plot
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            plots_data.append({
                'type': 'multivariate_summary',
                'title': 'PCA Projection 2D',
                'img': img_str
            })
            
        except Exception as e:
            print(f"PCA Plot Error: {e}")

        # 2. Time Series for each feature (for context)
        # We assume operational state is only for pairwise or is the first column if not provided
        # For multivariate, just plot the feature with outliers
        
        target_op_state = op_state if op_state else (plot_cols[0] if plot_cols else None) 

        for feat in plot_cols[:10]: # Limit to first 10 to avoid explosion
            
            fig_ts, ax_ts = plt.subplots(figsize=(10, 4))
            
            # Plot Feature (Blue) on Primary Y if no Op State, or Secondary Y if Op State exists
            # Requested: "no need to show the op state together with the features" for multivariate
            # So we strictly use single axis
            
            ax_ts.plot(plot_df[time_col], plot_df[feat], color='blue', alpha=0.5, label=feat, linewidth=1)
            
            # Highlight Outliers (Red) - These are global multivariate outliers
            if not outliers.empty:
                ax_ts.scatter(outliers[time_col], outliers[feat], color='red', s=20, label='Global Outlier', zorder=10, marker='x')
            
            ax_ts.set_ylabel(feat, color='blue')
            ax_ts.tick_params(axis='y', labelcolor='blue')
            
            ax_ts.set_title(f"Multivariate Context: {feat}")
            ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            # Legend - Bottom Outside
            ax_ts.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

            plt.tight_layout()
            
            # Save Time Series
            buf_ts = BytesIO()
            fig_ts.savefig(buf_ts, format="png", dpi=100)
            buf_ts.seek(0)
            img_ts = base64.b64encode(buf_ts.read()).decode('utf-8')
            plt.close(fig_ts)

            plots_data.append({
                'type': 'multivariate_ts',
                'title': feat,
                'ts_img': img_ts
            })

    return plots_data