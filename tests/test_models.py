"""Tests for neural models."""

import torch

from tabula_rasa import ProductionTableQA, StatisticalEncoder


class TestStatisticalEncoder:
    """Test suite for StatisticalEncoder."""

    def test_encoder_initialization(self):
        """Test encoder initialization."""
        encoder = StatisticalEncoder(hidden_dim=256, output_dim=768)

        assert encoder.hidden_dim == 256
        assert encoder.output_dim == 768

    def test_encoder_forward(self, sample_sketch):
        """Test encoder forward pass."""
        encoder = StatisticalEncoder(output_dim=512)
        output = encoder(sample_sketch)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (512,)

    def test_encoder_empty_sketch(self):
        """Test encoder with no numeric columns."""
        encoder = StatisticalEncoder(output_dim=512)
        sketch = {"columns": {}, "n_rows": 0, "n_cols": 0}

        output = encoder(sketch)
        assert torch.all(output == 0)


class TestProductionTableQA:
    """Test suite for ProductionTableQA model."""

    def test_model_initialization(self):
        """Test model initialization."""
        model = ProductionTableQA(model_name="t5-small", stat_dim=512)

        assert model.stat_encoder.output_dim == 512
        assert hasattr(model, "text_encoder")
        assert hasattr(model, "fusion")
        assert hasattr(model, "numeric_head")
        assert hasattr(model, "confidence_head")
        assert hasattr(model, "query_type_head")

    def test_model_forward(self, sample_sketch, qa_model):
        """Test model forward pass."""
        question = "What is the average price?"

        output = qa_model(question, sample_sketch)

        assert "answer" in output
        assert "confidence" in output
        assert "query_type_logits" in output

        assert isinstance(output["answer"], torch.Tensor)
        assert isinstance(output["confidence"], torch.Tensor)
        assert isinstance(output["query_type_logits"], torch.Tensor)

    def test_model_eval_mode(self, sample_sketch, qa_model):
        """Test model in eval mode."""
        qa_model.eval()
        question = "What is the maximum revenue?"

        with torch.no_grad():
            output = qa_model(question, sample_sketch)

        assert output["confidence"].item() >= 0.0
        assert output["confidence"].item() <= 1.0

    def test_model_return_features(self, sample_sketch, qa_model):
        """Test model with return_features=True."""
        question = "What is the sum of quantity?"

        output = qa_model(question, sample_sketch, return_features=True)

        assert "features" in output
        assert isinstance(output["features"], torch.Tensor)

    def test_parameter_count(self, qa_model):
        """Test model has expected number of parameters."""
        total_params = sum(p.numel() for p in qa_model.parameters())

        # t5-small has ~38M params (depending on variant), plus our additional layers
        # Expected range: 35M - 45M parameters
        assert 35_000_000 < total_params < 45_000_000
