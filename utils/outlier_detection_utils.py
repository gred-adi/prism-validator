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
import textwrap

# Import correlation function from model_dev_utils
from utils.model_dev_utils import corrfunc

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
    """Detects outliers in a feature series relative to an operational state.

    This function applies various outlier detection algorithms to a feature
    series, considering its relationship with an operational state variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        op_state_col (str): The name of the operational state column.
        feature (str): The name of the feature column to analyze.
        method (str, optional): The outlier detection method to use.
        contamination (float, optional): The expected proportion of outliers.
        n_neighbors (int, optional): The number of neighbors for LOF.
        iqr_factor (float, optional): The IQR factor for the IQR method.
        percentile_cut (float, optional): The percentile cutoff.
        residual_threshold (float, optional): The residual threshold for RANSAC.

    Returns:
        pd.Series: A boolean Series indicating outliers (True = Outlier).
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
    """Detects outliers in a multivariate dataset.

    This function applies outlier detection algorithms to the entire feature
    set simultaneously, identifying global anomalies.

    Args:
        df (pd.DataFrame): The input DataFrame.
        feature_cols (list): A list of feature column names.
        method (str, optional): The outlier detection method to use.
        contamination (float, optional): The expected proportion of outliers.

    Returns:
        pd.Series: A boolean Series indicating outliers (True = Outlier).
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
    """Generates visualizations for outlier detection results.

    This function creates a series of plots to visualize the detected outliers,
    with different plots generated based on whether a pairwise or multivariate
    strategy was used.

    Args:
        df_original (pd.DataFrame): The original DataFrame.
        mask_outliers (pd.Series): A boolean Series indicating outliers.
        strategy (str): The outlier detection strategy used ('pairwise' or 'multivariate').
        op_state (str, optional): The name of the operational state column.
        plot_cols (list, optional): A list of column names to plot.
        time_col (str, optional): The name of the time column.

    Returns:
        list: A list of dictionaries, where each dictionary contains plot
        details, including the plot type, title, and base64 encoded image data.
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
                # print(f"KDE Plot Error for {feat}: {e}")
                pass

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

        # 2. Time Series for each feature
        for feat in plot_cols[:10]: # Limit to first 10
            fig_ts, ax_ts = plt.subplots(figsize=(10, 4))
            
            ax_ts.plot(plot_df[time_col], plot_df[feat], color='blue', alpha=0.5, label=feat, linewidth=1)
            
            # Highlight Outliers (Red)
            if not outliers.empty:
                ax_ts.scatter(outliers[time_col], outliers[feat], color='red', s=20, label='Global Outlier', zorder=10, marker='x')
            
            ax_ts.set_ylabel(feat, color='blue')
            ax_ts.tick_params(axis='y', labelcolor='blue')
            ax_ts.set_title(f"Multivariate Context: {feat}")
            ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
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

def generate_correlation_analysis(df, numeric_cols):
    """Analyzes correlations in the data to recommend an outlier detection strategy.

    This function calculates the correlation matrix for the given numeric
    columns and recommends either a 'Pairwise' or 'Multivariate' outlier
    detection strategy based on the strength of the correlations.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numeric_cols (list): A list of numeric column names to analyze.

    Returns:
        Tuple[str, dict, str]: A tuple containing the recommended strategy,
        a dictionary of correlation statistics, and the reason for the
        recommendation.
    """
    if df is None or df.empty or not numeric_cols:
        return "Not enough data", {}, None

    # Calculate correlation matrix (absolute values to detect strength regardless of direction)
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Mask diagonal (correlation with self is always 1)
    mask = np.ones_like(corr_matrix, dtype=bool)
    np.fill_diagonal(mask, False)
    
    # Extract stats from off-diagonal elements
    off_diag_corr = corr_matrix.where(mask)
    avg_corr = off_diag_corr.mean().mean()
    max_corr = off_diag_corr.max().max()
    
    # Decision Logic
    # If there's strong linear relationship (high avg or some very high pairs), suggest Pairwise
    if avg_corr > 0.5 or max_corr > 0.8:
        recommendation = "Pairwise"
        reason = f"Strong linear relationships detected (Max Corr: {max_corr:.2f}, Avg Corr: {avg_corr:.2f}). Pairwise methods (like Residual/RANSAC) often work best here."
    else:
        recommendation = "Multivariate"
        reason = f"Weaker linear correlations detected (Max Corr: {max_corr:.2f}, Avg Corr: {avg_corr:.2f}). Multivariate methods (like Isolation Forest/PCA) are better for detecting structural anomalies in lower-correlation datasets."

    return recommendation, {"avg": avg_corr, "max": max_corr}, reason

def generate_pairplot_visuals(df, numeric_cols, title_suffix=""):
    """Generates a customized pairplot to visualize data relationships.

    This function creates a PairGrid plot with scatterplots on the lower
    triangle, correlation coefficients on the upper triangle, and kernel
    density estimates on the diagonal.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numeric_cols (list): A list of numeric column names to plot.
        title_suffix (str, optional): A suffix to add to the plot title.

    Returns:
        dict: A dictionary containing the base64 encoded pairplot image.
    """
    images = {}
    
    if not numeric_cols:
        return images

    # Downsample row count for performance
    MAX_POINTS = 800
    if len(df) > MAX_POINTS:
        plot_df = df[numeric_cols].sample(n=MAX_POINTS, random_state=42)
    else:
        plot_df = df[numeric_cols]
        
    # --- FIX: Wrap Labels for Readability ---
    wrapper = textwrap.TextWrapper(width=15, break_long_words=False)
    # Map original col names to wrapped names
    wrapped_cols = {col: "\n".join(wrapper.wrap(col)) for col in numeric_cols}
    plot_df = plot_df.rename(columns=wrapped_cols)
    
    # Determine plot height based on number of variables to fit on screen
    # Fewer vars = larger plots, More vars = smaller plots
    num_vars = len(numeric_cols)
    plot_height = 2.5 if num_vars < 10 else 2.0 
    
    try:
        g = sns.PairGrid(plot_df, height=plot_height, diag_sharey=False)
        g.map_upper(corrfunc, cmap=plt.get_cmap('BrBG'), norm=plt.Normalize(vmin=-1, vmax=1))
        g.map_lower(sns.scatterplot, s=50, color='#018571', alpha=0.6, edgecolor=None)
        g.map_diag(sns.kdeplot, color='red', fill=True, alpha=0.3)
        g.fig.subplots_adjust(wspace=0.1, hspace=0.1)
        
        # --- NEW: Improved Axis Label Handling ---
        for ax in g.axes.flatten():
            if ax:
                # 1. Rotate & Shrink Tick Labels
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha('right')
                    label.set_fontsize(7) 
                
                for label in ax.get_yticklabels():
                    label.set_fontsize(7)
                
                # 2. Adjust Axis Title Font Size (The wrapped Metric Name)
                xaxis = ax.get_xlabel()
                yaxis = ax.get_ylabel()
                if xaxis: ax.set_xlabel(xaxis, fontsize=9, labelpad=10) # Added padding
                if yaxis: ax.set_ylabel(yaxis, fontsize=9, labelpad=10)

        # Adjust main title position
        title = "Feature Relationships"
        if title_suffix:
            title += f" ({title_suffix})"
        
        g.fig.suptitle(title, y=1.02, fontsize=16)
        
        buf_pp = BytesIO()
        g.savefig(buf_pp, format="png", dpi=100, bbox_inches='tight')
        buf_pp.seek(0)
        images["pairplot"] = base64.b64encode(buf_pp.read()).decode('utf-8')
        plt.close(g.fig)
    except Exception as e:
        print(f"Pairplot generation failed: {e}")
    
    return images