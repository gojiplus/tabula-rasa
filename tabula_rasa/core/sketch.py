"""Statistical sketch extraction for compact table representation."""

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from sklearn.covariance import LedoitWolf


class AdvancedStatSketch:
    """
    Production-grade statistical sketch with:
    - Automatic distribution detection
    - Robust copula estimation
    - Conditional distribution inference
    - Multi-table relationship tracking
    """

    def __init__(self, max_categories: int = 50, confidence_level: float = 0.95):
        """
        Initialize the statistical sketch extractor.

        Args:
            max_categories: Maximum number of categorical values to track
            confidence_level: Confidence level for statistical estimates
        """
        self.max_cats = max_categories
        self.confidence = confidence_level

    def extract(self, df: pd.DataFrame, table_name: str = "table") -> dict:
        """
        Extract comprehensive statistical sketch from a DataFrame.

        Args:
            df: Input DataFrame
            table_name: Name identifier for the table

        Returns:
            Dictionary containing statistical sketch with columns, correlations,
            copula parameters, and conditional distributions
        """
        sketch = {
            "table_name": table_name,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": {},
            "correlations": {},
            "copula": None,
            "mutual_information": {},
            "conditional_distributions": {},
        }

        # Column-level statistics
        for col in df.columns:
            sketch["columns"][col] = self._extract_column_stats(df[col], col)

        # Pairwise relationships
        numeric_cols = [c for c, v in sketch["columns"].items() if v["type"] == "numeric"]

        if len(numeric_cols) > 1:
            # Spearman correlations (robust to non-linear)
            sketch["correlations"] = self._compute_robust_correlations(df[numeric_cols])

            # Gaussian copula for joint distribution
            sketch["copula"] = self._fit_gaussian_copula(df[numeric_cols])

            # Mutual information for non-linear dependencies
            sketch["mutual_information"] = self._estimate_mutual_information(df[numeric_cols])

        # Conditional distributions (for "given X, what is Y?" queries)
        sketch["conditional_distributions"] = self._extract_conditional_patterns(df)

        return sketch

    def _extract_column_stats(self, series: pd.Series, _col_name: str) -> dict:
        """Extract rich column statistics."""
        if pd.api.types.is_numeric_dtype(series):
            return self._numeric_column_stats(series)
        else:
            return self._categorical_column_stats(series)

    def _numeric_column_stats(self, series: pd.Series) -> dict:
        """Comprehensive numeric column statistics."""
        clean = series.dropna()

        if len(clean) == 0:
            return {"type": "numeric", "error": "no_data"}

        # Basic moments
        quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

        # Distribution shape
        stats = {
            "type": "numeric",
            "n_unique": len(clean.unique()),
            "mean": float(clean.mean()),
            "std": float(clean.std()),
            "min": float(clean.min()),
            "max": float(clean.max()),
            "quantiles": {q: float(clean.quantile(q)) for q in quantiles},
            "missing_rate": float(series.isna().mean()),
            "skewness": float(clean.skew()),
            "kurtosis": float(clean.kurtosis()),
        }

        # Outlier detection (IQR method)
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outlier_mask = (clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)
            stats["outlier_rate"] = float(outlier_mask.mean())
        else:
            stats["outlier_rate"] = 0.0

        # Distribution type hint (helps model understand data)
        stats["distribution_hint"] = self._detect_distribution_type(clean)

        return stats

    def _categorical_column_stats(self, series: pd.Series) -> dict:
        """Comprehensive categorical statistics."""
        value_counts = series.value_counts()
        total = len(series)

        # Top-k most frequent values
        top_k = min(self.max_cats, len(value_counts))
        top_values = value_counts.head(top_k).to_dict()

        # Diversity metrics
        stats = {
            "type": "categorical",
            "n_unique": len(value_counts),
            "top_values": {str(k): int(v) for k, v in top_values.items()},
            "missing_rate": float(series.isna().mean()),
            "mode": str(value_counts.index[0]),
            "mode_frequency": float(value_counts.iloc[0] / total),
        }

        # Shannon entropy (measure of diversity)
        probs = value_counts / total
        stats["entropy"] = float(-(probs * np.log(probs + 1e-10)).sum())
        stats["entropy_normalized"] = stats["entropy"] / np.log(len(value_counts) + 1e-10)

        # Gini coefficient (measure of concentration)
        sorted_counts = np.sort(value_counts.values)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        stats["gini"] = float(
            (2 * np.sum((np.arange(1, n + 1)) * sorted_counts)) / (n * cumsum[-1]) - (n + 1) / n
        )

        return stats

    def _detect_distribution_type(self, series: pd.Series) -> str:
        """Heuristic distribution type detection."""
        skew = series.skew()
        kurt = series.kurtosis()
        cv = series.std() / (abs(series.mean()) + 1e-10)  # Coefficient of variation

        # Simple heuristics
        if abs(skew) < 0.5 and abs(kurt) < 1:
            return "normal"
        elif skew > 1:
            return "right_skewed"
        elif skew < -1:
            return "left_skewed"
        elif abs(kurt) > 3:
            return "heavy_tailed"
        elif series.min() >= 0 and cv > 1:
            return "exponential"
        else:
            return "unknown"

    def _compute_robust_correlations(self, df: pd.DataFrame) -> dict:
        """Compute multiple correlation measures."""
        correlations = {}
        cols = df.columns

        for i, col1 in enumerate(cols):
            for col2 in cols[i + 1 :]:
                # Spearman (rank-based, robust)
                clean_data = df[[col1, col2]].dropna()
                if len(clean_data) < 3:
                    continue

                spearman_corr, _ = spearmanr(clean_data[col1], clean_data[col2])

                # Pearson (linear)
                pearson_corr = clean_data[col1].corr(clean_data[col2])

                if abs(spearman_corr) > 0.1 or abs(pearson_corr) > 0.1:
                    correlations[f"{col1}|{col2}"] = {
                        "spearman": float(spearman_corr),
                        "pearson": float(pearson_corr),
                        "strength": (
                            "strong"
                            if abs(spearman_corr) > 0.7
                            else "moderate"
                            if abs(spearman_corr) > 0.4
                            else "weak"
                        ),
                    }

        return correlations

    def _fit_gaussian_copula(self, df: pd.DataFrame) -> dict:
        """
        Fit Gaussian copula - captures dependency structure.

        Key insight: Copula separates margins from dependence
        """
        # Step 1: Rank-transform to uniform [0,1]
        uniform_data = df.apply(lambda x: rankdata(x, nan_policy="omit") / (len(x.dropna()) + 1))

        # Step 2: Transform to standard normal via inverse CDF
        normal_data = uniform_data.apply(lambda x: norm.ppf(np.clip(x, 0.001, 0.999)))

        # Step 3: Estimate correlation matrix with shrinkage (Ledoit-Wolf)
        clean_data = normal_data.dropna()
        if len(clean_data) < 2:
            return {"type": "gaussian", "error": "insufficient_data"}

        cov_estimator = LedoitWolf()
        cov_estimator.fit(clean_data)

        # Compute eigendecomposition for sampling
        eigenvalues, eigenvectors = np.linalg.eigh(cov_estimator.covariance_)

        return {
            "type": "gaussian",
            "covariance": cov_estimator.covariance_.tolist(),
            "shrinkage": float(cov_estimator.shrinkage_),
            "columns": df.columns.tolist(),
            "eigenvalues": eigenvalues.tolist(),
            "condition_number": float(eigenvalues.max() / (eigenvalues.min() + 1e-10)),
        }

    def _estimate_mutual_information(self, df: pd.DataFrame) -> dict:
        """Estimate mutual information (captures non-linear dependencies)."""
        mi_scores = {}

        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i + 1 :]:
                clean_data = df[[col1, col2]].dropna()
                if len(clean_data) < 10:
                    continue

                # Bin data and compute empirical MI
                mi = self._compute_mi_continuous(clean_data[col1].values, clean_data[col2].values)
                if mi > 0.1:
                    mi_scores[f"{col1}|{col2}"] = float(mi)

        return mi_scores

    def _compute_mi_continuous(self, x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
        """Compute mutual information for continuous variables."""
        try:
            # Discretize via equal-frequency binning
            x_binned = pd.qcut(x, bins, labels=False, duplicates="drop")
            y_binned = pd.qcut(y, bins, labels=False, duplicates="drop")

            # Compute MI via contingency table
            contingency = pd.crosstab(x_binned, y_binned)

            # Normalize
            pxy = contingency / contingency.sum().sum()
            px = pxy.sum(axis=1)
            py = pxy.sum(axis=0)

            # MI = sum p(x,y) log(p(x,y) / (p(x)p(y)))
            mi = 0.0
            for i in range(len(px)):
                for j in range(len(py)):
                    if pxy.iloc[i, j] > 0:
                        mi += pxy.iloc[i, j] * np.log(
                            pxy.iloc[i, j] / (px.iloc[i] * py.iloc[j] + 1e-10) + 1e-10
                        )

            return max(0, mi)
        except Exception:
            return 0.0

    def _extract_conditional_patterns(self, df: pd.DataFrame) -> dict:
        """Extract patterns for conditional queries (e.g., mean of X given Y > threshold)."""
        patterns = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for target_col in numeric_cols:
            for condition_col in numeric_cols:
                if target_col == condition_col:
                    continue

                # Compute conditional means at quartiles
                quartiles = df[condition_col].quantile([0.25, 0.5, 0.75])

                conditional_means = {}
                for q_name, q_val in zip(["Q1", "Q2", "Q3"], quartiles):
                    mask = df[condition_col] <= q_val
                    if mask.sum() > 10:  # Ensure sufficient samples
                        conditional_means[q_name] = float(df[mask][target_col].mean())

                if conditional_means:
                    patterns[f"{target_col}|{condition_col}"] = conditional_means

        return patterns
