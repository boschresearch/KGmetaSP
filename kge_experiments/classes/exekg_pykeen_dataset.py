# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


"""PyKEEN dataset class for ExeKGs with numeric literals support."""

import gc
import pathlib
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pykeen.datasets import NumericPathDataset
from pykeen.triples import TriplesFactory, TriplesNumericLiteralsFactory

from kge_experiments.config import (
    EXEKGS_NT_PATH,
    EXEKGS_W_MLSEAKG_NT_PATH,
    EXEKGS_MKGA_NT_PATH,
    EXEKGS_W_MLSEAKG_MKGA_NT_PATH,
)


class ExeKGPykeenDataset(NumericPathDataset):
    """PyKEEN dataset for ExeKGs with automatic numeric literals extraction."""

    training: TriplesNumericLiteralsFactory

    def __init__(
        self,
        use_mlseakg: bool = False,
        use_mkga: bool = False,
        create_inverse_triples: bool = True,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize ExeKG PyKEEN dataset.

        Args:
            use_mlseakg: Whether to include MLSeaKG data
            use_mkga: Whether to use MKGA preprocessed data
            create_inverse_triples: Whether to create inverse triples
            random_state: Random seed for train/test/val splits
            **kwargs: Additional arguments passed to parent class
        """
        self.use_mlseakg = use_mlseakg
        self.use_mkga = use_mkga
        self.random_state = random_state

        # Determine which .nt file to use
        if use_mlseakg:
            if use_mkga:
                self.nt_path = EXEKGS_W_MLSEAKG_MKGA_NT_PATH
            else:
                self.nt_path = EXEKGS_W_MLSEAKG_NT_PATH
        else:
            if use_mkga:
                self.nt_path = EXEKGS_MKGA_NT_PATH
            else:
                self.nt_path = EXEKGS_NT_PATH

        # Store parameters for lazy loading
        self._create_inverse_triples = create_inverse_triples
        self._triples_df = None
        self._numeric_literals_df = None

    def _load_nt_file(
        self, nt_path: pathlib.Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and parse .nt file into triples and numeric literals.

        Args:
            nt_path: Path to .nt file

        Returns:
            Tuple of (triples_df, numeric_literals_df)
        """
        triples = []
        numeric_literals = []

        print(f"Loading .nt file: {nt_path}")
        
        with open(nt_path, "r", encoding="utf-8") as f:
            for line in f:
                    
                if not line.strip():
                    continue

                # Parse triple: subject predicate object .
                # Replace the second space with tab for easier parsing
                line = re.sub(r" ", "\t", line, count=2)
                line = re.sub(r" \.$", "", line.strip())

                parts = line.split("\t")
                if len(parts) != 3:
                    continue

                subject, predicate, obj = parts

                # Remove angle brackets from URIs
                subject = subject.strip("<>")
                predicate = predicate.strip("<>")

                # Check if object is a numeric literal
                if self._is_numeric_literal(obj):
                    # Extract numeric value
                    numeric_value = self._extract_numeric_value(obj)
                    if numeric_value is not None:
                        numeric_literals.append(
                            {
                                "entity": subject,
                                "literal": predicate,
                                "value": numeric_value,
                            }
                        )
                else:
                    # Regular triple (object is an entity)
                    obj = obj.strip("<>").strip('"')
                    triples.append({"head": subject, "relation": predicate, "tail": obj})

        print(f"Loaded {len(triples):,} triples and {len(numeric_literals):,} numeric literals")
        
        triples_df = pd.DataFrame(triples)
        numeric_literals_df = pd.DataFrame(numeric_literals)

        # Free memory
        del triples
        del numeric_literals
        gc.collect()

        return triples_df, numeric_literals_df

    def _is_numeric_literal(self, obj: str) -> bool:
        """Check if object is a numeric literal (int or float)."""
        # Match patterns like "123"^^<http://www.w3.org/2001/XMLSchema#integer>
        # or "123.45"^^<http://www.w3.org/2001/XMLSchema#float>
        numeric_pattern = r'"([+-]?\d+\.?\d*)".*?(?:integer|float|double|decimal|int|long)'
        return bool(re.search(numeric_pattern, obj, re.IGNORECASE))

    def _extract_numeric_value(self, obj: str) -> Optional[float]:
        """Extract numeric value from literal string."""
        # Extract the value between quotes
        match = re.search(r'"([+-]?\d+\.?\d*)"', obj)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _split_triples(
        self, triples_df: pd.DataFrame, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split triples into train/test/validation (80/10/10).

        Args:
            triples_df: DataFrame with columns [head, relation, tail]
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, test_df, val_df)
        """
        # Shuffle the dataframe
        shuffled_df = triples_df.sample(frac=1, random_state=random_state).reset_index(
            drop=True
        )

        n = len(shuffled_df)
        train_size = int(0.8 * n)
        test_size = int(0.1 * n)

        train_df = shuffled_df[:train_size]
        test_df = shuffled_df[train_size : train_size + test_size]
        val_df = shuffled_df[train_size + test_size :]

        return train_df, test_df, val_df

    def _create_triples_factory(
        self,
        triples_df: pd.DataFrame,
        numeric_literals_df: pd.DataFrame,
        create_inverse_triples: bool = False,
        entity_to_id: Optional[dict] = None,
    ):
        """
        Create TriplesFactory from dataframes.

        Args:
            triples_df: DataFrame with columns [head, relation, tail]
            numeric_literals_df: DataFrame with columns [entity, literal, value]
            create_inverse_triples: Whether to create inverse triples
            entity_to_id: Optional existing entity_to_id mapping

        Returns:
            TriplesNumericLiteralsFactory if numeric literals present, else TriplesFactory
        """
        # Convert triples to numpy array
        triples_array = triples_df[["head", "relation", "tail"]].values

        # Prepare numeric_triples - shape (n, 3): entity, attribute_relation, attribute_value
        if len(numeric_literals_df) > 0:
            numeric_triples = numeric_literals_df[["entity", "literal", "value"]].values
            
            # Create the factory with numeric literals
            factory = TriplesNumericLiteralsFactory.from_labeled_triples(
                triples=triples_array,
                create_inverse_triples=create_inverse_triples,
                entity_to_id=entity_to_id,
                numeric_triples=numeric_triples,
                literal_matrix_preprocessing="minmax",
            )
        else:
            # Create regular factory without numeric literals for test/val
            factory = TriplesFactory.from_labeled_triples(
                triples=triples_array,
                create_inverse_triples=create_inverse_triples,
                entity_to_id=entity_to_id,
            )

        return factory

    def _load(self) -> None:
        """Load training and testing triples factories."""
        # Load and parse the .nt file
        if self._triples_df is None:
            self._triples_df, self._numeric_literals_df = self._load_nt_file(
                self.nt_path
            )

        # Split into train/test/val (80/10/10)
        train_df, test_df, val_df = self._split_triples(
            self._triples_df, self.random_state
        )

        # Create training factory with all numeric literals
        # (PyKEEN will filter to only entities present in training)
        print(f"Creating training factory with {len(train_df):,} triples...")
        self._training = self._create_triples_factory(
            train_df,
            self._numeric_literals_df,
            create_inverse_triples=self._create_inverse_triples,
        )

        # Free memory - we don't need the full triples_df anymore
        del train_df
        del self._triples_df
        del self._numeric_literals_df
        self._triples_df = None
        self._numeric_literals_df = None
        gc.collect()
        print(f"Training factory created. Memory freed.")

        # For test/val, pass empty numeric literals to save memory
        # They will use the same entity_to_id mapping from training
        empty_literals_df = pd.DataFrame(columns=["entity", "literal", "value"])
        
        # Create testing factory
        print(f"Creating testing factory with {len(test_df):,} triples...")
        self._testing = self._create_triples_factory(
            test_df,
            empty_literals_df,
            entity_to_id=self._training.entity_to_id,
        )
        
        del test_df
        gc.collect()
        print(f"Testing factory created. Memory freed.")

        # Store validation split for later
        self._val_df = val_df
        self._empty_literals_df = empty_literals_df

    def _load_validation(self) -> None:
        """Load validation triples factory."""
        # Create validation factory using stored val_df with empty literals
        print(f"Creating validation factory with {len(self._val_df):,} triples...")
        self._validation = self._create_triples_factory(
            self._val_df,
            self._empty_literals_df,
            entity_to_id=self._training.entity_to_id,
        )
        
        # Free memory
        del self._val_df
        del self._empty_literals_df
        self._val_df = None
        self._empty_literals_df = None
        gc.collect()
        print(f"Validation factory created. Dataset fully loaded.")


if __name__ == "__main__":
    # Test the dataset
    dataset = ExeKGPykeenDataset(use_mlseakg=False, use_mkga=False)
    dataset.summarize()
