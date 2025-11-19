"""Neural encoders for statistical sketches and text."""

import torch
import torch.nn as nn


class StatisticalEncoder(nn.Module):
    """
    Encode statistical sketch into neural representation.

    Handles variable-length column lists via attention pooling.
    """

    def __init__(self, hidden_dim: int = 256, output_dim: int = 768):
        """
        Initialize the statistical encoder.

        Args:
            hidden_dim: Hidden dimension for column encoders
            output_dim: Output dimension of the encoded sketch
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Column-level encoders
        self.numeric_encoder = nn.Sequential(
            nn.Linear(15, hidden_dim),  # 15 numeric features per column
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

        # Attention pooling over columns
        self.attention = nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)

        # Global table encoder
        self.table_encoder = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, sketch: dict) -> torch.Tensor:
        """
        Encode sketch to fixed-size vector.

        Args:
            sketch: Statistical sketch dictionary

        Returns:
            Tensor of shape (output_dim,) representing the encoded table
        """
        column_embeddings = []

        # Encode each numeric column
        for _col_name, col_stats in sketch["columns"].items():
            if col_stats["type"] == "numeric" and "error" not in col_stats:
                # Pack statistics into feature vector
                features = torch.tensor(
                    [
                        col_stats["mean"],
                        col_stats["std"],
                        col_stats["min"],
                        col_stats["max"],
                        col_stats["quantiles"][0.25],
                        col_stats["quantiles"][0.5],
                        col_stats["quantiles"][0.75],
                        col_stats["skewness"],
                        col_stats["kurtosis"],
                        col_stats["missing_rate"],
                        col_stats["outlier_rate"],
                        col_stats["n_unique"] / max(sketch["n_rows"], 1),  # Normalized
                        1.0 if col_stats["distribution_hint"] == "normal" else 0.0,
                        1.0 if col_stats["distribution_hint"] == "right_skewed" else 0.0,
                        1.0 if col_stats["distribution_hint"] == "heavy_tailed" else 0.0,
                    ],
                    dtype=torch.float32,
                )

                embedding = self.numeric_encoder(features)
                column_embeddings.append(embedding)

        if not column_embeddings:
            # No numeric columns - return zero vector
            return torch.zeros(self.output_dim)

        # Stack column embeddings
        column_stack = torch.stack(column_embeddings).unsqueeze(0)  # (1, n_cols, dim)

        # Attention pooling
        attended, _ = self.attention(column_stack, column_stack, column_stack)
        pooled = attended.mean(dim=1).squeeze(0)  # (dim,)

        # Final encoding
        return self.table_encoder(pooled)
