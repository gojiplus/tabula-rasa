"""Production Table QA model with T5 backbone."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .encoders import StatisticalEncoder


class ProductionTableQA(nn.Module):
    """
    Production Table QA model with T5 backbone.

    Combines:
    - Pretrained language understanding (T5)
    - Statistical table knowledge (StatEncoder)
    - Execution grounding (trained to match executor)
    """

    def __init__(self, model_name: str = "t5-small", stat_dim: int = 768):
        """
        Initialize the Table QA model.

        Args:
            model_name: Pretrained T5 model name
            stat_dim: Dimension for statistical encoder output
        """
        super().__init__()

        # T5 encoder for question understanding
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        t5_model = AutoModel.from_pretrained(model_name)
        self.text_encoder = t5_model.encoder
        self.text_config = t5_model.config

        # Statistical sketch encoder
        self.stat_encoder = StatisticalEncoder(output_dim=stat_dim)

        # Fusion layer (combine text + stats)
        text_dim = self.text_config.d_model
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + stat_dim, stat_dim),
            nn.LayerNorm(stat_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(stat_dim, stat_dim),
            nn.LayerNorm(stat_dim),
            nn.ReLU(),
        )

        # Output heads
        self.numeric_head = nn.Sequential(
            nn.Linear(stat_dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(stat_dim, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
        )

        # Query type classifier (helps model route questions)
        self.query_type_head = nn.Sequential(
            nn.Linear(stat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # aggregate, conditional, filter, group_by
        )

    def forward(
        self, question: str, sketch: dict, return_features: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            question: Natural language question
            sketch: Statistical sketch dictionary
            return_features: Whether to return intermediate features

        Returns:
            Dictionary with keys:
                - answer: Predicted numerical answer
                - confidence: Confidence score [0, 1]
                - query_type_logits: Query type classification logits
                - features (optional): Fused representation
        """
        # Encode question with T5
        inputs = self.tokenizer(
            question, return_tensors="pt", padding=True, truncation=True, max_length=128
        )

        text_outputs = self.text_encoder(**inputs)
        text_repr = text_outputs.last_hidden_state.mean(dim=1)  # Mean pooling

        # Encode table statistics
        stat_repr = self.stat_encoder(sketch).unsqueeze(0)  # Add batch dim

        # Fuse
        combined = torch.cat([text_repr, stat_repr], dim=-1)
        fused = self.fusion(combined)

        # Predictions
        numeric_answer = self.numeric_head(fused).squeeze()
        confidence = self.confidence_head(fused).squeeze()
        query_type_logits = self.query_type_head(fused).squeeze()

        output = {
            "answer": numeric_answer,
            "confidence": confidence,
            "query_type_logits": query_type_logits,
        }

        if return_features:
            output["features"] = fused

        return output
