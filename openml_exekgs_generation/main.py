# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from typing import List
import typer

from openml_exekgs_generation.classes.openml_crawler import OpenMLCrawler

app = typer.Typer()


@app.command()
def runs_to_exekgs(
    mp_runs: bool = False,
    mp_tasks: bool = False,
    task_id: List[int] = None,
    use_processed_tasks_and_runs: bool = False,
    offset_step: int = 1000,
    n_tasks: int = None,
    n_runs_per_flow: int = 10,
):
    crawler = OpenMLCrawler(
        mp_runs,
        mp_tasks,
        task_id,
        use_processed_tasks_and_runs,
        offset_step,
        n_tasks,
        n_runs_per_flow,
    )

    crawler.crawl_runs()

if __name__ == "__main__":
    app()
