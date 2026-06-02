# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
import re
from typing import Tuple


def get_openml_ids_from_file_path(file_path: str) -> Tuple[int, int, int]:
    sep = os.path.sep
    if sep == "\\":
        sep = "\\\\"
    match = re.search(rf"task_(\d+){sep}flow_(\d+){sep}run_(\d+)\.ttl", file_path)
    task_id = match.group(1)
    flow_id = match.group(2)
    run_id = match.group(3)

    return int(task_id), int(flow_id), int(run_id)
