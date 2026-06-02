# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


"""Link Prediction trainer using PyKEEN."""

import json
import pathlib
from typing import Any, Dict, Optional

import torch
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator

from kge_experiments.classes.exekg_pykeen_dataset import ExeKGPykeenDataset
from kge_experiments.config import PYKEEN_LP_HPO_CONFIG


class LinkPredictionTrainer:
    """Trainer class for link prediction models."""

    SUPPORTED_MODELS = [
        "DistMultReaLitEGated",
        "TransEReaLitEGated",
        "DistMultReaLitE",
        "TransEReaLitE",
        "ComplExReaLitEGated",
        "ComplExReaLitE",
    ]

    def __init__(
        self,
        model_type: str,
        use_mlseakg: bool = False,
        use_mkga: bool = False,
        create_inverse_triples: bool = True,
        timeout_hours: Optional[float] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True,
    ):
        """
        Initialize the link prediction trainer.

        Args:
            model_type: Type of model to train (DistMultReaLitEGated, etc.)
            use_mlseakg: Whether to use MLSeaKG data
            use_mkga: Whether to use MKGA preprocessing
            create_inverse_triples: Whether to create inverse triples
            timeout_hours: Timeout in hours for training
            device: Device to use for training ('cuda' or 'cpu')
            verbose: Whether to print progress
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model {model_type} not supported. Choose from {self.SUPPORTED_MODELS}"
            )

        self.model_type = model_type
        self.use_mlseakg = use_mlseakg
        self.use_mkga = use_mkga
        self.create_inverse_triples = create_inverse_triples
        self.timeout_hours = timeout_hours
        self.device = device
        self.verbose = verbose

        # Load dataset
        if self.verbose:
            print(f"Loading ExeKG dataset (mlseakg={use_mlseakg}, mkga={use_mkga})...")
        self.dataset = ExeKGPykeenDataset(
            use_mlseakg=use_mlseakg,
            use_mkga=use_mkga,
            create_inverse_triples=create_inverse_triples,
        )

        # Load configuration
        self.hpo_config = PYKEEN_LP_HPO_CONFIG

    def train(self, save_dir: pathlib.Path) -> Dict[str, Any]:
        """
        Train model using default parameters.

        Args:
            save_dir: Directory to save trained model and results

        Returns:
            Dictionary containing training results and metrics
        """
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"Starting training for {self.model_type}...")
            print(f"Device: {self.device}")
            if self.timeout_hours:
                print(f"Timeout: {self.timeout_hours} hours")

        # Get model-specific default configuration
        model_kwargs = self.hpo_config["pipeline"]["model_to_model_kwargs"].get(
            self.model_type, {}
        )

        # Get training configuration with defaults
        stopper_kwargs = self.hpo_config["pipeline"]["stopper_kwargs"]
        training_kwargs = self.hpo_config["pipeline"]["training_kwargs"]

        # Set hyperparameters based on dataset size
        # For large datasets (>500K entities, >5M triples):
        # - Lower embedding dim to save memory
        # - Larger batch size for efficiency
        # - Lower dropout (large dataset = less overfitting)
        # - More negative samples for better training signal
        
        default_model_kwargs = model_kwargs.copy()
        default_model_kwargs["input_dropout"] = 0.1  # Lower dropout for large dataset
        default_model_kwargs["embedding_dim"] = 128  # Moderate size for memory efficiency
        
        # Batch size: larger for efficiency, but not too large to fit in memory
        # With 600K entities and batch_size=2048, memory usage is manageable
        batch_size = 2048
        
        # Learning rate: slightly lower for stability with large batches
        learning_rate = 0.0005
        
        # Negative samples: more samples improve training but increase memory
        num_negs_per_pos = 3
        
        # Loss margin: higher for larger datasets to push entities apart
        margin = 50

        default_training_kwargs = training_kwargs.copy()
        default_training_kwargs["batch_size"] = batch_size
        
        if self.verbose:
            print(f"\nHyperparameters:")
            print(f"  Embedding dim: {default_model_kwargs['embedding_dim']}")
            print(f"  Dropout: {default_model_kwargs['input_dropout']}")
            print(f"  Batch size: {batch_size}")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Neg samples per pos: {num_negs_per_pos}")
            print(f"  Margin: {margin}")

        # Run pipeline
        result = pipeline(
            device=self.device,
            model=self.model_type,
            dataset=self.dataset,
            epochs=1500,
            stopper=self.hpo_config["pipeline"]["stopper"],
            stopper_kwargs=stopper_kwargs,
            model_kwargs=default_model_kwargs,
            loss="SelfAdversarialNegativeSampling",
            loss_kwargs=dict(adversarial_temperature=1.0, margin=margin),
            training_kwargs=default_training_kwargs,
            optimizer=self.hpo_config["pipeline"]["optimizer"],
            optimizer_kwargs=dict(lr=learning_rate),
            negative_sampler="BasicNegativeSampler",
            negative_sampler_kwargs=dict(num_negs_per_pos=num_negs_per_pos),
            training_loop="sLCWA",
            training_loop_kwargs=dict(
                negative_sampler="BasicNegativeSampler",
            ),
            evaluator=RankBasedEvaluator,
        )

        if self.verbose:
            print(f"\nTraining completed!")

        # Save model
        result.save_to_directory(save_dir)

        # Evaluate on test set
        test_results = self._evaluate_on_test(result, save_dir)

        # Save embeddings
        self._save_embeddings(result, save_dir)

        results = {
            "model_type": self.model_type,
            "use_mlseakg": self.use_mlseakg,
            "use_mkga": self.use_mkga,
            "create_inverse_triples": self.create_inverse_triples,
            "test_results": test_results,
        }

        # Save overall results
        with open(save_dir / "training_results.json", "w") as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            print(f"\nResults saved to {save_dir}")

        return results

    def _evaluate_on_test(
        self, pipeline_result, save_dir: pathlib.Path
    ) -> Dict[str, float]:
        """Evaluate model on test set."""
        if self.verbose:
            print("\nEvaluating model on test set...")

        # Evaluate on test set
        test_results = pipeline_result.model.evaluate(
            mapped_triples=self.dataset.testing.mapped_triples,
            batch_size=256,
            use_tqdm=self.verbose,
        )

        # Convert metrics to dict
        metrics = {
            "mrr": float(test_results.get_metric("mean_reciprocal_rank")),
            "hits_at_1": float(test_results.get_metric("hits_at_1")),
            "hits_at_3": float(test_results.get_metric("hits_at_3")),
            "hits_at_10": float(test_results.get_metric("hits_at_10")),
        }

        if self.verbose:
            print("\nTest Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        # Save test results
        with open(save_dir / "test_results.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def _save_embeddings(self, pipeline_result, save_dir: pathlib.Path):
        """Save entity and relation embeddings."""
        import pickle

        best_model = pipeline_result.model

        # Extract embeddings
        entity_embeddings = (
            best_model.entity_representations[0](indices=None).detach().cpu().numpy()
        )
        relation_embeddings = (
            best_model.relation_representations[0](indices=None).detach().cpu().numpy()
        )

        # Save embeddings
        embeddings_data = {
            "entity_embeddings": entity_embeddings,
            "relation_embeddings": relation_embeddings,
            "entity_to_id": self.dataset.training.entity_to_id,
            "relation_to_id": self.dataset.training.relation_to_id,
        }

        with open(save_dir / "embeddings.pkl", "wb") as f:
            pickle.dump(embeddings_data, f)

        if self.verbose:
            print(f"\nEmbeddings saved to {save_dir / 'embeddings.pkl'}")
            print(f"  Entity embeddings shape: {entity_embeddings.shape}")
            print(f"  Relation embeddings shape: {relation_embeddings.shape}")


if __name__ == "__main__":
    # Example usage
    trainer = LinkPredictionTrainer(
        model_type="DistMultReaLitEGated",
        use_mlseakg=False,
        use_mkga=False,
    )

    save_path = pathlib.Path("data/kge_models/pykeen_lp/test_run")
    results = trainer.train(save_path)
    print(json.dumps(results, indent=2))
