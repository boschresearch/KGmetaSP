# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


"""CLI entry point."""


import pathlib
from kge_experiments.constants.filter_by_datasets import FILTER_BY_DATASET_IDS
from kge_experiments.classes.dataset_similarity_calculator import (
    DatasetSimilarityCalculator,
)
import typer

from kge_experiments.classes.exekg_converter import ExeKGConverter
from kge_experiments.classes.rdf2vec import RDF2Vec
from kge_experiments.config import (
    EXEKGS_MKGA_NT_PATH,
    EXEKGS_TGZ_PATH,
    EXEKGS_W_MLSEAKG_MKGA_NT_PATH,
    EXEKGS_W_MLSEAKG_NT_PATH,
    EXEKGS_NT_PATH,
    EXEKGS_W_MLSEAKG_TGZ_PATH,
    MKGA_CONFIG_RELPATH,
    MLSEAKG_FILTERED_NT_PATH,
    RDF2VEC_MODEL_DIR,
)
from kge_experiments.constants.invalid_flows_per_task import INVALID_FLOWS_PER_TASK
from kge_experiments.utils.graph_utils import combine_exekgs_nt_with_mlseakg_nt, filter_mlseakg
from kge_experiments.utils.kgbench_utils import (
    create_kgbench_dataset_from_nt,
    create_nt_from_kgbench_data,
)
from kge_experiments.utils.openml_utils import read_task_info_from_logs
from kge_experiments.classes.pipeline_performance_predictor import PipelinePerformancePredictor

from hydra import initialize, compose

import importlib

mkga_exekgs_extension = importlib.import_module("MKGA-exekgs-extension.src.main")
preprocess_data = getattr(mkga_exekgs_extension, "preprocess_data")

app = typer.Typer(
    name="cli", help="CLI entry point.", no_args_is_help=True, add_completion=False
)


# Global variable to store verbose state
verbose = False
rdf2vec_walk_distance = 10
rdf2vec_num_walks = 10
rdf2vec_walk_strategy = "random"
use_mlseakg = False
use_mkga = False


def set_verbose(ctx: typer.Context, param: typer.CallbackParam, value: bool):
    global verbose
    verbose = value


def set_rdf2vec_walk_distance(
    ctx: typer.Context, param: typer.CallbackParam, value: int
):
    global rdf2vec_walk_distance
    rdf2vec_walk_distance = value


def set_rdf2vec_num_walks(ctx: typer.Context, param: typer.CallbackParam, value: int):
    global rdf2vec_num_walks
    rdf2vec_num_walks = value


def set_rdf2vec_walk_strategy(
    ctx: typer.Context, param: typer.CallbackParam, value: str
):
    global rdf2vec_walk_strategy
    rdf2vec_walk_strategy = value


def set_use_mlseakg(ctx: typer.Context, param: typer.CallbackParam, value: bool):
    global use_mlseakg
    use_mlseakg = value


def set_use_mkga(ctx: typer.Context, param: typer.CallbackParam, value: bool):
    global use_mkga
    use_mkga = value


@app.callback()
def main(
    verbose: bool = typer.Option(
        False, "--verbose", help="Enable verbose mode", callback=set_verbose
    ),
    rdf2vec_d: int = typer.Option(
        10, help="The distance of the walks", callback=set_rdf2vec_walk_distance
    ),
    rdf2vec_w: int = typer.Option(
        10, help="The number of walks", callback=set_rdf2vec_num_walks
    ),
    rdf2vec_ws: str = typer.Option(
        "random",
        help="The walk strategy",
        callback=set_rdf2vec_walk_strategy,
    ),
    use_mlseakg: bool = typer.Option(
        False, help="Use MLSeaKG in addition to ExeKGs", callback=set_use_mlseakg
    ),
    use_mkga: bool = typer.Option(
        False, help="Use MKGA preprocessing method", callback=set_use_mkga
    ),
):
    pass


@app.command()
def calculate_kge_similarities(
    replace_similarities: bool = False,
    replace_info: bool = False,
    add_data_entity_sim: bool = False,
    add_data_entity_and_pipeline_sim: bool = False,
    add_pipeline_sim: bool = False,
    add_dataset_info: bool = False,
):
    if not any(
        [
            add_data_entity_sim,
            add_data_entity_and_pipeline_sim,
            add_pipeline_sim,
            add_dataset_info,
        ]
    ):
        typer.echo(
            "You need to specify at least one of the following options: add-data-entity-sim, add-data-entity-and-pipeline-sim, add-pipeline-sim, add-dataset-info"
        )
        raise typer.Exit(code=1)

    DatasetSimilarityCalculator(
        verbose,
        rdf2vec_walk_distance,
        rdf2vec_num_walks,
        rdf2vec_walk_strategy,
        use_mlseakg,
        use_mkga,
        FILTER_BY_DATASET_IDS,
        INVALID_FLOWS_PER_TASK,
    ).calculate_kge_similarities(
        replace_similarities,
        replace_info,
        add_data_entity_sim,
        add_data_entity_and_pipeline_sim,
        add_pipeline_sim,
        add_dataset_info,
    )


@app.command()
def calculate_and_add_geds_to_similarities(
    replace_geds: bool = False,
    add_dataset_info: bool = False,
    replace_info: bool = False,
    num_processes: int = 10,
    chunk_size: int = 5,
):
    DatasetSimilarityCalculator(
        verbose,
        filter_by_dataset_ids=FILTER_BY_DATASET_IDS,
        exclude_flows_per_task=INVALID_FLOWS_PER_TASK,
    ).calculate_and_add_geds_to_similarities(
        replace_geds, add_dataset_info, replace_info, num_processes, chunk_size
    )

@app.command()
def prepare_data(
    filter_by_dataset_ids: bool = False,
    excl_flows_per_task: bool = False,
    excl_performance_values: bool = False,
    remove_old_files: bool = False,
):
    """
    Converts (filtered) ExeKGs from .ttl to .nt format and concatenates them into a single file.
    """

    if remove_old_files:
        EXEKGS_NT_PATH.unlink(missing_ok=True)
        MLSEAKG_FILTERED_NT_PATH.unlink(missing_ok=True)
        EXEKGS_W_MLSEAKG_NT_PATH.unlink(missing_ok=True)
        EXEKGS_TGZ_PATH.unlink(missing_ok=True)
        EXEKGS_W_MLSEAKG_TGZ_PATH.unlink(missing_ok=True)
        EXEKGS_MKGA_NT_PATH.unlink(missing_ok=True)
        EXEKGS_W_MLSEAKG_MKGA_NT_PATH.unlink(missing_ok=True)

    if remove_old_files or not EXEKGS_NT_PATH.exists():
        ExeKGConverter(
            filter_by_dataset_ids=(
                FILTER_BY_DATASET_IDS if filter_by_dataset_ids else None
            ),
            exclude_flows_per_task=(
                INVALID_FLOWS_PER_TASK if excl_flows_per_task else None
            ),
            exclude_performance_values=excl_performance_values,
        ).run()

    if filter_by_dataset_ids:
        dataset_ids = FILTER_BY_DATASET_IDS
    else:
        task_info_df = read_task_info_from_logs()
        dataset_ids = task_info_df["dataset_id"].astype(int).unique().tolist()

    if remove_old_files or not MLSEAKG_FILTERED_NT_PATH.exists():
        filter_mlseakg(dataset_ids)
    if remove_old_files or not EXEKGS_W_MLSEAKG_NT_PATH.exists():
        combine_exekgs_nt_with_mlseakg_nt(dataset_ids)

    if remove_old_files or not EXEKGS_TGZ_PATH.exists():
        EXEKGS_TGZ_PATH.parent.mkdir(parents=True, exist_ok=True)
        create_kgbench_dataset_from_nt(EXEKGS_NT_PATH, EXEKGS_TGZ_PATH)

    if remove_old_files or not EXEKGS_W_MLSEAKG_TGZ_PATH.exists():
        EXEKGS_W_MLSEAKG_TGZ_PATH.parent.mkdir(parents=True, exist_ok=True)
        create_kgbench_dataset_from_nt(
            EXEKGS_W_MLSEAKG_NT_PATH, EXEKGS_W_MLSEAKG_TGZ_PATH
        )

    if remove_old_files or not EXEKGS_MKGA_NT_PATH.exists():
        # exit(0)
        with initialize(config_path=str(MKGA_CONFIG_RELPATH.parent)):
            cfg = compose(
                MKGA_CONFIG_RELPATH.name.split(".")[0],
                overrides=["pipeline.dataload=exekgs"],
            )
            kgbench_data = preprocess_data(cfg)
            print("Preprocessed ExeKGs using MKGA")
            create_nt_from_kgbench_data(kgbench_data, EXEKGS_MKGA_NT_PATH)

    if remove_old_files or not EXEKGS_W_MLSEAKG_MKGA_NT_PATH.exists():
        with initialize(config_path=str(MKGA_CONFIG_RELPATH.parent)):
            cfg = compose(
                config_name=MKGA_CONFIG_RELPATH.name.split(".")[0],
                overrides=["pipeline.dataload=exekgs_with_mlseakg"],
            )
            kgbench_data = preprocess_data(cfg)
            print("Preprocessed ExeKGs+MLSeaKG using MKGA")
            create_nt_from_kgbench_data(kgbench_data, EXEKGS_W_MLSEAKG_MKGA_NT_PATH)


@app.command()
def train_rdf2vec(
    chunk_size: int = 100,
    cpu_count: int = 4,
):
    """
    Generates RDF2Vec embeddings from a knowledge graph.
    """

    print(f"Training RDF2Vec model with the following parameters:")
    print(f"Use MLSeaKG: {use_mlseakg}")
    print(f"Use MKGA: {use_mkga}")
    print(f"Walk distance: {rdf2vec_walk_distance}")
    print(f"Number of walks: {rdf2vec_num_walks}")
    print(f"Walk strategy: {rdf2vec_walk_strategy}")
    print(f"Chunk size: {chunk_size}")
    print(f"CPU count: {cpu_count}")

    if use_mlseakg:
        if use_mkga:
            data_path = str(EXEKGS_W_MLSEAKG_MKGA_NT_PATH)
        else:
            data_path = str(EXEKGS_W_MLSEAKG_NT_PATH)
    else:
        if use_mkga:
            data_path = str(EXEKGS_MKGA_NT_PATH)
        else:
            data_path = str(EXEKGS_NT_PATH)

    RDF2Vec(
        data_path=data_path,
        distance=rdf2vec_walk_distance,
        max_walks=rdf2vec_num_walks,
        train=True,
        chunksize=chunk_size,
        save_path=str(RDF2VEC_MODEL_DIR).format(
            d=rdf2vec_walk_distance,
            w=rdf2vec_num_walks,
            ws=rdf2vec_walk_strategy,
            kg=("exekgs+mlseakg" if use_mlseakg else "exekgs")
            + ("_mkga" if use_mkga else ""),
        ),
        cpu_count=cpu_count,
        walk_strategy=rdf2vec_walk_strategy,
    )


@app.command()
def predict_pipeline_performance(
    split_mode: str = typer.Option(
        "dataset", help="Split strategy - 'dataset' or 'pipeline'"
    ),
    target: str = typer.Option("f1_score", help="Target variable to predict"),
    emb_aggr_type: str = typer.Option(
        "concat", help="Embedding aggregation type - 'concat' or 'mean'"
    ),
    model: str = typer.Option("RF", help="Model type - RF, SVC, LR, RFReg, SVR, LRReg"),
    dataset_emb_source: str = typer.Option(
        "rdf2vec", help="Dataset embedding source - 'rdf2vec', 'pykeen-lp', or metafeature types"
    ),
    pipeline_emb_source: str = typer.Option(
        "rdf2vec", help="Pipeline embedding source - 'rdf2vec', 'pykeen-lp'"
    ),
    min_train_samples_per_run_id: int = typer.Option(
        50, help="Minimum training samples per run ID"
    ),
    pykeen_lp_model_name: str = typer.Option(
        None, help="PyKEEN LP model name (e.g., 'DistMultReaLitEGated', 'TransEReaLitE'). Path will be constructed automatically."
    ),
    pykeen_lp_inverse_triples: bool = typer.Option(
        True, help="Whether the PyKEEN model was trained with inverse triples (used for path construction)"
    ),
):
    """Run pipeline performance prediction using embeddings and machine learning."""
    
    # Validate pykeen-lp requirements and construct path
    pykeen_lp_model_path = None
    if pipeline_emb_source == "pykeen-lp" or dataset_emb_source == "pykeen-lp":
        if pykeen_lp_model_name is None:
            typer.echo("Error: --pykeen-lp-model-name is required when using pykeen-lp for dataset or pipeline embeddings")
            raise typer.Exit(code=1)
        
        # Construct the model path dynamically
        from kge_experiments.config import PYKEEN_LP_MODEL_PATH
        pykeen_lp_model_path = pathlib.Path(
            str(PYKEEN_LP_MODEL_PATH).format(
                model_type=pykeen_lp_model_name,
                use_mlseakg=use_mlseakg,
                use_mkga=use_mkga,
                inv_tri=pykeen_lp_inverse_triples,
            )
        )
        
        if verbose:
            typer.echo(f"Using PyKEEN LP model: {pykeen_lp_model_name}")
            typer.echo(f"Dataset embeddings: {dataset_emb_source}")
            typer.echo(f"Pipeline embeddings: {pipeline_emb_source}")
            typer.echo(f"Model path: {pykeen_lp_model_path}")
        
        # Verify the path exists
        if not pykeen_lp_model_path.exists():
            typer.echo(f"Error: PyKEEN LP model path does not exist: {pykeen_lp_model_path}")
            typer.echo(f"Please ensure the model has been trained with the specified parameters:")
            typer.echo(f"  Model: {pykeen_lp_model_name}")
            typer.echo(f"  MLSeaKG: {use_mlseakg}")
            typer.echo(f"  MKGA: {use_mkga}")
            typer.echo(f"  Inverse triples: {pykeen_lp_inverse_triples}")
            raise typer.Exit(code=1)
    
    predictor = PipelinePerformancePredictor(
        split_mode=split_mode,
        target=target,
        emb_aggr_type=emb_aggr_type,
        model=model,
        dataset_emb_source=dataset_emb_source,
        pipeline_emb_source=pipeline_emb_source,
        use_mlseakg=use_mlseakg,
        use_mkga=use_mkga,
        walk_distance=rdf2vec_walk_distance,
        num_walks=rdf2vec_num_walks,
        min_train_samples_per_run_id=min_train_samples_per_run_id,
        verbose=verbose,
        pykeen_lp_model_path=str(pykeen_lp_model_path) if pykeen_lp_model_path else None,
    )

    predictor.run_prediction()


@app.command()
def train_link_prediction(
    model_type: str = typer.Option(
        "DistMultReaLitEGated",
        help="Model type: DistMultReaLitEGated, TransEReaLitEGated, DistMultReaLitE, TransEReaLitE, ComplExReaLitEGated, ComplExReaLitE",
    ),
    timeout_hours: float = typer.Option(None, help="Timeout in hours"),
    inverse_triples: bool = typer.Option(True, help="Create inverse triples"),
):
    """Train link prediction model."""
    from kge_experiments.classes.link_prediction_trainer import LinkPredictionTrainer
    from kge_experiments.config import PYKEEN_LP_MODEL_PATH

    if verbose:
        typer.echo(f"Training {model_type}...")
        typer.echo(f"Use MLSeaKG: {use_mlseakg}")
        typer.echo(f"Use MKGA: {use_mkga}")
        typer.echo(f"Inverse triples: {inverse_triples}")

    # Create trainer
    trainer = LinkPredictionTrainer(
        model_type=model_type,
        use_mlseakg=use_mlseakg,
        use_mkga=use_mkga,
        create_inverse_triples=inverse_triples,
        timeout_hours=timeout_hours,
        verbose=verbose,
    )

    # Determine save directory
    save_dir = pathlib.Path(
        str(PYKEEN_LP_MODEL_PATH).format(
            model_type=model_type,
            use_mlseakg=use_mlseakg,
            use_mkga=use_mkga,
            inv_tri=inverse_triples,
        )
    )

    # Train model
    results = trainer.train(save_dir)

    if verbose:
        typer.echo(f"\nTraining complete!")
        typer.echo(f"Best MRR: {results['test_results']['mrr']:.4f}")
        typer.echo(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    app()
