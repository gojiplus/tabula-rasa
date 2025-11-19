"""Production training pipeline with best practices."""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.qa_model import ProductionTableQA
from .dataset import TableQADataset


class ProductionTrainer:
    """Production training with best practices."""

    def __init__(
        self,
        model: ProductionTableQA,
        df: pd.DataFrame,
        sketch: dict,
        lr: float = 1e-4,
        batch_size: int = 16,
        device: str = "cpu",
    ):
        """
        Initialize the trainer.

        Args:
            model: ProductionTableQA model to train
            df: Training DataFrame
            sketch: Statistical sketch of the DataFrame
            lr: Learning rate
            batch_size: Batch size for training
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.df = df
        self.sketch = sketch
        self.batch_size = batch_size
        self.device = device

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        # Loss weights
        self.answer_weight = 1.0
        self.confidence_weight = 0.1
        self.query_type_weight = 0.2

    def train(
        self, n_epochs: int = 10, n_train_samples: int = 1000, n_val_samples: int = 200
    ) -> tuple[float, dict]:
        """
        Training loop with validation.

        Args:
            n_epochs: Number of training epochs
            n_train_samples: Number of training samples to generate
            n_val_samples: Number of validation samples to generate

        Returns:
            Tuple of (best_val_loss, history_dict)
        """
        # Create datasets
        train_dataset = TableQADataset(self.df, self.sketch, n_train_samples)
        val_dataset = TableQADataset(self.df, self.sketch, n_val_samples)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("Warning: No valid samples generated")
            return float("inf"), {}

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_mape": []}

        for epoch in range(n_epochs):
            # Training
            train_loss = self._train_epoch(train_loader)

            # Validation
            val_loss, val_metrics = self._validate(val_loader)

            # Scheduler step
            self.scheduler.step(val_loss)

            # Store history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_mae"].append(val_metrics["mae"])
            history["val_mape"].append(val_metrics["mape"])

            # Logging
            if epoch % 2 == 0:
                print(f"Epoch {epoch}/{n_epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val MAE: {val_metrics['mae']:.4f}")
                print(f"  Val MAPE: {val_metrics['mape']:.2%}")
                print(f"  Query Type Acc: {val_metrics['query_type_acc']:.2%}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        return best_val_loss, history

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Single training epoch."""
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            questions = batch["question"]
            true_answers = batch["answer"]
            query_types = batch["query_type"]

            # Forward pass for each question in batch
            batch_loss = 0
            for q, true_ans, qt in zip(questions, true_answers, query_types):
                output = self.model(q, self.sketch)

                # Convert tensor to float if needed
                true_ans_val = true_ans.item() if torch.is_tensor(true_ans) else float(true_ans)

                # Skip if answer is invalid
                if np.isnan(true_ans_val) or np.isinf(true_ans_val):
                    continue

                # Answer loss (MSE) with normalization
                true_ans_tensor = torch.tensor(true_ans_val, dtype=torch.float32).to(self.device)
                answer_loss = F.mse_loss(output["answer"], true_ans_tensor) / (
                    abs(true_ans_val) + 1.0
                )

                # Confidence calibration
                with torch.no_grad():
                    error = abs(output["answer"].item() - true_ans_val)
                    target_conf = max(0, 1 - error / (abs(true_ans_val) + 1e-6))
                conf_loss = F.mse_loss(
                    output["confidence"],
                    torch.tensor(target_conf, dtype=torch.float32).to(self.device),
                )

                # Query type classification
                qt_loss = F.cross_entropy(
                    output["query_type_logits"].unsqueeze(0), qt.unsqueeze(0).to(self.device)
                )

                # Combined loss
                loss = (
                    self.answer_weight * answer_loss
                    + self.confidence_weight * conf_loss
                    + self.query_type_weight * qt_loss
                )

                batch_loss += loss

            if batch_loss > 0:
                # Backward
                self.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += batch_loss.item()

        return total_loss / max(len(dataloader), 1)

    def _validate(self, dataloader: DataLoader) -> tuple[float, dict]:
        """Validation."""
        self.model.eval()
        total_loss = 0
        errors = []
        true_values = []
        query_type_correct = 0
        query_type_total = 0

        with torch.no_grad():
            for batch in dataloader:
                questions = batch["question"]
                true_answers = batch["answer"]
                query_types = batch["query_type"]

                for q, true_ans, qt in zip(questions, true_answers, query_types):
                    # Convert tensor to float if needed
                    true_ans_val = true_ans.item() if torch.is_tensor(true_ans) else float(true_ans)

                    # Skip invalid answers
                    if np.isnan(true_ans_val) or np.isinf(true_ans_val):
                        continue

                    output = self.model(q, self.sketch)

                    # Loss
                    true_ans_tensor = torch.tensor(true_ans_val, dtype=torch.float32).to(
                        self.device
                    )
                    loss = F.mse_loss(output["answer"], true_ans_tensor)
                    total_loss += loss.item()

                    # Metrics
                    pred_val = output["answer"].item()
                    if not (np.isnan(pred_val) or np.isinf(pred_val)):
                        error = abs(pred_val - true_ans_val)
                        errors.append(error)
                        true_values.append(abs(true_ans_val))

                    # Query type accuracy
                    pred_qt = output["query_type_logits"].argmax().item()
                    if pred_qt == qt.item():
                        query_type_correct += 1
                    query_type_total += 1

        mae = np.mean(errors) if errors else 0.0

        # Calculate MAPE carefully
        if errors and true_values:
            mape = np.mean([e / max(tv, 1e-6) for e, tv in zip(errors, true_values)])
        else:
            mape = 0.0

        metrics = {
            "mae": mae,
            "mape": mape,
            "query_type_acc": query_type_correct / max(query_type_total, 1),
        }

        return total_loss / max(len(dataloader), 1), metrics
