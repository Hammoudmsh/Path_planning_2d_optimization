#!/usr/bin/env python
# coding: utf-8

# # A*, JPS, and Smoothness-Aware JPS (SA-JPS) Benchmark
# 
# This notebook benchmarks unidirectional and bidirectional grid-search approaches on MovingAI `.map` and `.scen` datasets:
# 
# 1. **A\*** — optimal baseline.
# 2. **Standard JPS** — distance-optimal symmetry-pruned search.
# 3. **Smooth JPS** — distance remains primary; turns are only an equal-distance tie-breaker.
# 4. **SA-JPS** — direction-aware JPS minimizing a composite distance-and-turn objective during search.
# 5. **Fast Weighted SA-JPS** — goal-directed SA-JPS using a weighted heuristic.
# 6. **JPS–Dijkstra Corridor** — experimental JPS-guided local Dijkstra refinement.
# 
# The notebook downloads the MovingAI street dataset, runs selected scenarios, validates every path, and exports per-run, summary, paired-comparison, parameter-sweep, and plotting results.
# 
# > **Research note:** SA-JPS is an experimental extension. Standard JPS pruning was derived for distance-based uniform grids. Adding heading-dependent turn costs changes the objective, so empirical validation against a direction-aware A* reference is included for rigorous evaluation.
# 
# 
# Additional experimental approaches:
# - **Bidirectional Standard JPS**
# - **Bidirectional SA-JPS**
# - **Bidirectional Fast Weighted SA-JPS**
# 
# 
# Configuration is now selected from `benchmark_config.py` using a dataset name and experiment profile.
# 

# In[1]:


# Run once if your environment is missing packages.
# %pip install -q numpy pandas matplotlib tqdm scipy scikit-learn


# In[1]:


from __future__ import annotations

import argparse
from benchmark_resume import SearchJournal, reuse_astar_references
import sys
import re
from pathlib import Path

from benchmark_config import (
    DATASETS,
    EXPERIMENT_CONFIGS,
    get_config,
)
import sys
sys.setrecursionlimit(30000)
# --------------------------------------------------------------
# JUPYTER DEFAULTS
# Edit only these two values when running inside Jupyter.
# --------------------------------------------------------------
DEFAULT_DATASET = "street"
DEFAULT_CONFIG = "debug"
RESULTS_DIR = ""
from collections import Counter
from pathlib import Path
from typing import Optional, Union, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


def generate_dataset_description_table(
    *,
    config: Optional[Dict[str, object]] = None,
    map_paths: Optional[Sequence[Path]] = None,
    count_selected_scenarios: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "dataset_description",
    display_table: bool = True,
) -> pd.DataFrame:
    """
    Generate a dataset-description table for the explicitly supplied
    configuration.

    The function does not depend on global MAPS_DIR, SCENARIOS_DIR,
    RESULTS_DIR, or the currently active notebook dataset.
    """

    active_config = (
        dict(config)
        if config is not None
        else dict(CONFIG)
    )

    dataset_name = str(
        active_config.get(
            "dataset_name",
            "dataset",
        )
    ).strip()

    maps_dir = Path(
        active_config["maps_dir"]
    )

    scenarios_dir = Path(
        active_config["scenarios_dir"]
    )

    results_dir = Path(
        active_config["results"]
    )

    if not maps_dir.exists():
        raise FileNotFoundError(
            f"Maps directory does not exist for dataset "
            f"{dataset_name!r}: {maps_dir.resolve()}"
        )

    if not scenarios_dir.exists():
        raise FileNotFoundError(
            f"Scenarios directory does not exist for dataset "
            f"{dataset_name!r}: {scenarios_dir.resolve()}"
        )

    # ----------------------------------------------------------
    # Discover maps from this configuration only
    # ----------------------------------------------------------
    if map_paths is None:
        all_map_paths = sorted(
            maps_dir.rglob("*.map")
        )

        requested_sizes = active_config.get(
            "map_sizes"
        )
        print(requested_sizes)
        if requested_sizes:
            requested_sizes = {
                int(size)
                for size in requested_sizes
            }

            filtered_maps = []

            for map_path in all_map_paths:
                try:
                    grid = read_map(map_path)
                except Exception as error:
                    print(
                        f"Skipping unreadable map "
                        f"{map_path.name}: {error}"
                    )
                    continue

                height, width = grid.shape

                if (
                    width in requested_sizes
                    or height in requested_sizes
                ):
                    filtered_maps.append(map_path)

            maps = filtered_maps
        else:
            maps = all_map_paths

        map_limit = active_config.get(
            "map_limit"
        )

        if map_limit is not None:
            maps = maps[: int(map_limit)]

    else:
        maps = [
            Path(path)
            for path in map_paths
        ]

    if not maps:
        raise FileNotFoundError(
            f"No maps were found for dataset {dataset_name!r} "
            f"under {maps_dir.resolve()}."
        )

    # ----------------------------------------------------------
    # Local scenario-file resolver
    # ----------------------------------------------------------
    def find_scenario_file(
        map_path: Path,
    ) -> Path:
        expected_name = (
            map_path.name + ".scen"
        )

        matches = list(
            scenarios_dir.rglob(
                expected_name
            )
        )

        if matches:
            return sorted(matches)[0]

        # Some datasets use map stem without .map.
        alternative_name = (
            map_path.stem + ".scen"
        )

        alternative_matches = list(
            scenarios_dir.rglob(
                alternative_name
            )
        )

        if alternative_matches:
            return sorted(
                alternative_matches
            )[0]

        raise FileNotFoundError(
            f"No scenario file was found for "
            f"{map_path.name!r} under "
            f"{scenarios_dir.resolve()}."
        )

    # ----------------------------------------------------------
    # Analyse every map
    # ----------------------------------------------------------
    map_records: List[
        Dict[str, object]
    ] = []

    for map_path in maps:
        grid = read_map(
            map_path
        )

        if grid.ndim != 2:
            raise ValueError(
                f"Map {map_path.name!r} is not "
                f"a two-dimensional grid."
            )

        height, width = grid.shape

        map_type, map_features = (
            get_map_type(grid)
        )

        scenario_path = (
            find_scenario_file(
                map_path
            )
        )

        scenarios = read_scenario(
            scenario_path
        )

        required_columns = [
            "start_x",
            "start_y",
            "goal_x",
            "goal_y",
        ]

        existing_required_columns = [
            column
            for column in required_columns
            if column in scenarios.columns
        ]

        if existing_required_columns:
            scenarios = scenarios.dropna(
                subset=existing_required_columns
            )

        available_scenario_count = int(
            len(scenarios)
        )

        if count_selected_scenarios:
            scenarios = select_scenarios(
                scenarios,
                active_config.get(
                    "scenarios_per_map"
                ),
                str(
                    active_config.get(
                        "scenario_selection",
                        "uniform",
                    )
                ),
            )

        selected_scenario_count = int(
            len(scenarios)
        )

        map_records.append(
            {
                "Dataset": dataset_name,
                "Map name": map_path.name,
                "Map type": str(map_type),
                "Width": int(width),
                "Height": int(height),
                "Resolution": (
                    f"{width} × {height}"
                ),
                "Available scenarios": (
                    available_scenario_count
                ),
                "Selected scenarios": (
                    selected_scenario_count
                ),
                "Scenario count": (
                    selected_scenario_count
                    if count_selected_scenarios
                    else available_scenario_count
                ),
                "Obstacle density": float(
                    map_features.get(
                        "Obstacle density",
                        np.nan,
                    )
                ),
                "Corridor ratio": float(
                    map_features.get(
                        "Corridor ratio",
                        np.nan,
                    )
                ),
                "Mean axis visibility": float(
                    map_features.get(
                        "Mean axis visibility",
                        np.nan,
                    )
                ),
            }
        )

    map_details_df = pd.DataFrame(
        map_records
    )

    # ----------------------------------------------------------
    # Aggregate map types
    # ----------------------------------------------------------
    type_counts = (
        map_details_df["Map type"]
        .value_counts()
        .sort_index()
    )

    map_type_description = "; ".join(
        f"{map_type}: {int(count)}"
        for map_type, count
        in type_counts.items()
    )

    # ----------------------------------------------------------
    # Aggregate resolutions
    # ----------------------------------------------------------
    resolution_counts = (
        map_details_df["Resolution"]
        .value_counts()
        .sort_index()
    )

    resolution_description = "; ".join(
        f"{resolution}: {int(count)}"
        for resolution, count
        in resolution_counts.items()
    )

    summary_df = pd.DataFrame(
        [
            {
                "Dataset": dataset_name,
                "Map type": (
                    map_type_description
                ),
                "#Maps": int(
                    len(map_details_df)
                ),
                "#Scenarios": int(
                    map_details_df[
                        "Scenario count"
                    ].sum()
                ),
                "Resolution": (
                    resolution_description
                ),
            }
        ]
    )

    # ----------------------------------------------------------
    # Dataset-specific output directory and filenames
    # ----------------------------------------------------------
    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else results_dir / "tables"
    )

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_dataset_name = (
        dataset_name
        .replace(" ", "_")
        .replace("/", "_")
    )

    output_stem = (
        f"{filename}_{safe_dataset_name}"
    )

    csv_path = (
        target_dir
        / f"{output_stem}.csv"
    )

    xlsx_path = (
        target_dir
        / f"{output_stem}.xlsx"
    )

    markdown_path = (
        target_dir
        / f"{output_stem}.md"
    )

    latex_path = (
        target_dir
        / f"{output_stem}.tex"
    )

    map_details_path = (
        target_dir
        / f"{output_stem}_per_map.csv"
    )

    summary_df.to_csv(
        csv_path,
        index=False,
    )

    summary_df.to_excel(
        xlsx_path,
        index=False,
    )

    summary_df.to_markdown(
        markdown_path,
        index=False,
    )

    latex_text = (
        summary_df.to_latex(
            index=False,
            escape=True,
            caption=(
                f"Description of the "
                f"{dataset_name} benchmark dataset."
            ),
            label=(
                f"tab:dataset_"
                f"{safe_dataset_name}"
            ),
        )
    )

    latex_path.write_text(
        latex_text,
        encoding="utf-8",
    )

    map_details_df.to_csv(
        map_details_path,
        index=False,
    )

    print(
        f"Dataset analysed : {dataset_name}"
    )
    print(
        f"Maps directory   : "
        f"{maps_dir.resolve()}"
    )
    print(
        f"Scenarios dir    : "
        f"{scenarios_dir.resolve()}"
    )
    print(
        f"Maps found       : "
        f"{len(map_details_df)}"
    )
    print(
        f"Scenarios counted: "
        f"{int(map_details_df['Scenario count'].sum())}"
    )
    print(
        f"Saved table      : "
        f"{csv_path.resolve()}"
    )

    if display_table:
        try:
            from IPython.display import (
                display,
            )

            display(summary_df)

        except ImportError:
            print(
                summary_df.to_string(
                    index=False
                )
            )

    return summary_df


def parse_configuration_selection():
    parser = argparse.ArgumentParser(
        description="A*, JPS, SA-JPS benchmark framework"
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        choices=sorted(DATASETS),
        default=DEFAULT_DATASET,
        help="MovingAI dataset name.",
    )

    parser.add_argument(
        "config",
        nargs="?",
        choices=sorted(EXPERIMENT_CONFIGS),
        default=DEFAULT_CONFIG,
        help="Experiment profile name.",
    )

    parser.add_argument("--config-json", type=Path, help="JSON experiment overrides; search code is unchanged.")
    parser.add_argument("--dry-run", action="store_true", help="Print the effective configuration and exit.")

    if "ipykernel" in sys.modules:
        # Jupyter injects unrelated kernel arguments. Use the defaults above.
        return parser.parse_args([])

    return parser.parse_args()


ARGS = parse_configuration_selection()

CONFIG = get_config(
    dataset_name=ARGS.dataset,
    profile_name=ARGS.config,
    project_root=Path.cwd(),
)

# Optional JSON changes configuration only, using the existing labeled dispatcher.
if ARGS.config_json is not None:
    import json
    experiment = json.loads(ARGS.config_json.read_text(encoding="utf-8-sig"))
    CONFIG = get_config(ARGS.dataset, experiment.get("base_profile", ARGS.config), project_root=Path.cwd())
    overrides = experiment.get("overrides", {})
    protected = {"dataset_name", "config_name", "project_root", "data_dir", "maps_dir",
                 "scenarios_dir", "results", "benchmark_cache_dir"}
    if protected.intersection(overrides):
        raise ValueError("Use output_root to change the result destination; dataset paths come from the original profile.")
    original_results = Path(CONFIG["results"]).resolve()
    output = Path.cwd() / experiment["output_root"] / ARGS.dataset / CONFIG["config_name"]
    if output.resolve() == original_results:
        raise ValueError("Use a separate output_root to preserve original results.")
    CONFIG.update(overrides)
    CONFIG.update(results=output, benchmark_cache_dir=output / "cache",
                  archive_name=f"{ARGS.dataset}_weighted_jps_results")

if ARGS.dry_run:
    import json
    print(json.dumps(CONFIG, indent=2, default=str))
    raise SystemExit(0)

print("=" * 72)
print("SELECTED EXPERIMENT")
print("=" * 72)
print(f"Dataset       : {CONFIG['dataset_name']}")
print(f"Configuration : {CONFIG['config_name']}")
print(f"Description   : {CONFIG.get('description', '')}")
print(f"Results       : {CONFIG['results']}")
print("=" * 72)

# ==============================================================
# DATASET-AWARE OUTPUT AND PLOT-DATA HELPERS
# ==============================================================

def active_dataset_name() -> str:
    """Return a filesystem-safe active dataset identifier."""
    value = str(globals().get("CONFIG", {}).get("dataset_name", "dataset"))
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip().lower())
    return value.strip("._-") or "dataset"


def dataset_output_path(
    path: Union[str, Path],
    *,
    dataset_name: Optional[str] = None,
) -> Path:
    """
    Insert the dataset name immediately before the file extension.

    Examples
    --------
    summary_results.csv -> summary_results_street.csv
    runtime.png         -> runtime_street.png

    Existing matching suffixes are not duplicated.
    """
    path = Path(path)
    token = dataset_name or active_dataset_name()
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(token).strip().lower())
    token = token.strip("._-") or "dataset"

    if path.suffix:
        if path.stem.lower().endswith(f"_{token.lower()}"):
            return path
        return path.with_name(f"{path.stem}_{token}{path.suffix}")

    if path.name.lower().endswith(f"_{token.lower()}"):
        return path
    return path.with_name(f"{path.name}_{token}")


def save_plot_data_csv(
    frame: pd.DataFrame,
    figure_path: Union[str, Path],
    *,
    label: str = "plot_data",
) -> Path:
    """Save the exact tabular data used to create a figure."""
    figure_path = Path(figure_path)
    raw_csv = figure_path.with_name(
        f"{figure_path.stem}_{label}.csv"
    )
    csv_path = dataset_output_path(raw_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    return csv_path


# ## Mean, median, and standard deviation outputs
# 
# All dataset-level, per-map, map-type, and ablation summary CSV files now contain the mean, median, and sample standard deviation for every available numeric evaluation criterion. Plot-data CSV files also include the standard-deviation column associated with each plotted mean or median metric.
# 

# In[2]:



import json
import hashlib

import heapq
import math
import os
import re
import time
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
# import matplotlib
# matplotlib.use('tkagg')  # Set backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        """
        Fallback for normal Python execution.
        """
        if hasattr(obj, "to_string"):
            print(obj.to_string())
        else:
            print(obj)

SQRT2 = math.sqrt(2.0)
Point = Tuple[int, int]       # (x, y)
Direction = Tuple[int, int]

DATA_DIR = Path(CONFIG["data_dir"])
MAPS_DIR = Path(CONFIG["maps_dir"])
SCENARIOS_DIR = Path(CONFIG["scenarios_dir"])
RESULTS_DIR = Path(CONFIG["results"])
CACHE_DIR = Path(CONFIG["benchmark_cache_dir"])

for directory in (
    DATA_DIR,
    MAPS_DIR,
    SCENARIOS_DIR,
    RESULTS_DIR,
    CACHE_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# ## 1. Configuration
# 
# For a quick test, keep `MAP_LIMIT` and `SCENARIOS_PER_MAP` small. Set `MAP_LIMIT=None` to run all selected maps.
# 
# **Movement rule:** to preserve compatibility with the uploaded implementation, a diagonal is forbidden only when both adjacent orthogonal cells are blocked. MovingAI reference lengths normally use stricter no-corner-cutting, so the notebook also compares every method against the internally optimal A* result.

# In[3]:


# Configuration is intentionally placed in the final notebook section.
# Edit only the final CONFIG cell, then run the final execution cell.


# ## 2. Download and read MovingAI data

# In[4]:


def download_and_extract(url: str, destination: Path) -> None:
    """Download a ZIP file and extract it if the destination is empty."""
    

    if any(destination.iterdir()):
        print(f'Skipping download: {destination} is not empty.')
        return

    archive_path = destination.parent / f'{destination.name}.zip'
    print(f'Downloading {url}')
    urllib.request.urlretrieve(url, archive_path)
    with zipfile.ZipFile(archive_path, 'r') as archive:
        archive.extractall(destination)
    archive_path.unlink(missing_ok=True)
    print(f'Extracted into {destination.resolve()}')


# In[5]:


def locate_file(root: Path, filename: str) -> Path:
    """Find a file recursively because ZIP layouts can differ."""
    direct = root / filename
    if direct.exists():
        return direct
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f'{filename!r} was not found under {root.resolve()}')
    return matches[0]


def read_map(filename: str | Path) -> np.ndarray:
    """
    Return a boolean grid with shape (height, width).
    True means traversable; False means blocked.
    """
    path = Path(filename)
    if not path.exists():
        path = locate_file(MAPS_DIR, str(filename))

    with path.open('r', encoding='utf-8') as handle:
        header = [handle.readline().strip() for _ in range(4)]
        height = int(header[1].split()[1])
        width = int(header[2].split()[1])
        rows = [handle.readline().rstrip('\n') for _ in range(height)]

    if len(rows) != height or any(len(row) != width for row in rows):
        raise ValueError(f'Invalid map dimensions in {path}')

    # MovingAI traversable terrain commonly includes '.', 'G', and 'S'.
    traversable = {'.', 'G', 'S'}
    return np.array([[cell in traversable for cell in row] for row in rows], dtype=bool)


def read_scenario(filename: str | Path) -> pd.DataFrame:
    path = Path(filename)
    if not path.exists():
        path = locate_file(SCENARIOS_DIR, str(filename))

    columns = [
        'bucket', 'map_name', 'width', 'height',
        'start_x', 'start_y', 'goal_x', 'goal_y', 'reference_length'
    ]
    return pd.read_csv(path, sep=r'\s+', header=None, names=columns, skiprows=1)


def obstacle_percentage(grid: np.ndarray) -> float:
    return 100.0 * float(np.count_nonzero(~grid)) / float(grid.size)


# ## 3. Shared geometry, path metrics, and result structure

# In[6]:


DIRECTIONS_8: Tuple[Direction, ...] = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)
DIRECTION_TO_ID = {direction: index for index, direction in enumerate(DIRECTIONS_8)}
START_DIRECTION_ID = 8


def sign(value: int) -> int:
    return (value > 0) - (value < 0)


def movement_direction(a: Point, b: Point) -> Direction:
    return sign(b[0] - a[0]), sign(b[1] - a[1])


def octile_distance(a: Point, b: Point) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (SQRT2 - 1.0) * min(dx, dy)


def segment_distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def in_bounds(grid: np.ndarray, x: int, y: int) -> bool:
    return 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]


def walkable(grid: np.ndarray, x: int, y: int) -> bool:
    return in_bounds(grid, x, y) and bool(grid[y, x])


def can_step(grid: np.ndarray, x: int, y: int, dx: int, dy: int) -> bool:
    nx, ny = x + dx, y + dy
    if not walkable(grid, nx, ny):
        return False
    # Match the original uploaded implementation: diagonal movement is blocked
    # only when both adjacent orthogonal cells are blocked.
    if dx != 0 and dy != 0:
        return walkable(grid, x + dx, y) or walkable(grid, x, y + dy)
    return True


def expand_jump_path(jump_path: Sequence[Point]) -> List[Point]:
    if not jump_path:
        return []
    full_path = [jump_path[0]]
    for start, end in zip(jump_path[:-1], jump_path[1:]):
        dx, dy = movement_direction(start, end)
        x, y = start
        while (x, y) != end:
            x += dx
            y += dy
            full_path.append((x, y))
    return full_path


def path_length(path: Sequence[Point]) -> float:
    return sum(segment_distance(a, b) for a, b in zip(path[:-1], path[1:]))


def path_turn_metrics(path: Sequence[Point]) -> Tuple[int, float, float]:
    """Return turn count, cumulative absolute turning angle, and maximum angle."""
    if len(path) < 3:
        return 0, 0.0, 0.0

    compressed = [path[0]]
    last_direction = None
    for previous, current in zip(path[:-1], path[1:]):
        direction = movement_direction(previous, current)
        if direction != last_direction:
            if compressed[-1] != previous:
                compressed.append(previous)
            last_direction = direction
    if compressed[-1] != path[-1]:
        compressed.append(path[-1])

    angles = []
    for a, b, c in zip(compressed[:-2], compressed[1:-1], compressed[2:]):
        v1 = np.array([b[0] - a[0], b[1] - a[1]], dtype=float)
        v2 = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
        cosine = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        cosine = max(-1.0, min(1.0, cosine))
        angle = math.degrees(math.acos(cosine))
        if angle > 1e-9:
            angles.append(angle)

    return len(angles), float(sum(angles)), float(max(angles, default=0.0))


def count_waypoints(path: Sequence[Point]) -> int:
    if not path:
        return 0
    turns, _, _ = path_turn_metrics(path)
    return turns + 2 if len(path) > 1 else 1


@dataclass
class SearchResult:
    algorithm: str
    success: bool
    path: List[Point]
    runtime_ms: float
    expanded_nodes: int
    generated_successors: int
    queue_pushes: int
    jump_calls: int = 0
    note: str = ''

    def metrics(self) -> Dict[str, float | int | bool | str]:
        turns, total_angle, max_angle = path_turn_metrics(self.path)
        return {
            'Algorithm': self.algorithm,
            'Success': self.success,
            'Path length': path_length(self.path) if self.success else np.nan,
            'Full path nodes': len(self.path),
            'Waypoints': count_waypoints(self.path),
            'Turns': turns,
            'Total turning angle deg': total_angle,
            'Maximum turn angle deg': max_angle,
            'Expanded nodes': self.expanded_nodes,
            'Generated successors': self.generated_successors,
            'Queue pushes': self.queue_pushes,
            'Jump calls': self.jump_calls,
            'Time ms': self.runtime_ms,
            'Note': self.note,
        }


# ## 4. A* baseline
# 
# The implementation uses the same movement rules as the JPS variants, making the comparison internally consistent.

# In[7]:


def reconstruct_point_path(parent: Dict[Point, Point], goal: Point) -> List[Point]:
    path = [goal]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def run_astar(grid: np.ndarray, start: Point, goal: Point) -> SearchResult:
    started = time.perf_counter()
    queue: List[Tuple[float, float, int, Point]] = []
    counter = 0
    heapq.heappush(queue, (octile_distance(start, goal), 0.0, counter, start))

    g_score: Dict[Point, float] = {start: 0.0}
    parent: Dict[Point, Point] = {}
    closed: Set[Point] = set()
    expanded = generated = pushes = 0

    while queue:
        _, queued_g, _, current = heapq.heappop(queue)
        if current in closed or not math.isclose(queued_g, g_score.get(current, math.inf), abs_tol=1e-12):
            continue
        closed.add(current)
        expanded += 1

        if current == goal:
            runtime = (time.perf_counter() - started) * 1000.0
            return SearchResult('A*', True, reconstruct_point_path(parent, goal), runtime,
                                expanded, generated, pushes)

        x, y = current
        for dx, dy in DIRECTIONS_8:
            if not can_step(grid, x, y, dx, dy):
                continue
            generated += 1
            neighbour = (x + dx, y + dy)
            tentative = g_score[current] + (SQRT2 if dx and dy else 1.0)
            if tentative + 1e-12 < g_score.get(neighbour, math.inf):
                g_score[neighbour] = tentative
                parent[neighbour] = current
                counter += 1
                heapq.heappush(queue, (
                    tentative + octile_distance(neighbour, goal),
                    tentative,
                    counter,
                    neighbour,
                ))
                pushes += 1

    runtime = (time.perf_counter() - started) * 1000.0
    return SearchResult('A*', False, [], runtime, expanded, generated, pushes, note='No path')


# ## 5. JPS successor generation
# 
# The jump function is cached within each search. This avoids repeating identical directional scans during the same query.

# In[8]:


def has_forced_neighbour(grid: np.ndarray, x: int, y: int, dx: int, dy: int) -> bool:
    if dx != 0 and dy != 0:
        return (
            (not walkable(grid, x - dx, y) and walkable(grid, x - dx, y + dy))
            or
            (not walkable(grid, x, y - dy) and walkable(grid, x + dx, y - dy))
        )
    if dx != 0:
        return (
            (not walkable(grid, x, y + 1) and walkable(grid, x + dx, y + 1))
            or
            (not walkable(grid, x, y - 1) and walkable(grid, x + dx, y - 1))
        )
    return (
        (not walkable(grid, x + 1, y) and walkable(grid, x + 1, y + dy))
        or
        (not walkable(grid, x - 1, y) and walkable(grid, x - 1, y + dy))
    )


def pruned_directions(grid: np.ndarray, current: Point, parent: Optional[Point]) -> List[Direction]:
    x, y = current
    if parent is None:
        return [(dx, dy) for dx, dy in DIRECTIONS_8 if can_step(grid, x, y, dx, dy)]

    dx, dy = movement_direction(parent, current)
    candidates: List[Direction] = []

    if dx != 0 and dy != 0:
        for direction in ((dx, dy), (dx, 0), (0, dy)):
            if can_step(grid, x, y, *direction):
                candidates.append(direction)
        if not walkable(grid, x - dx, y) and can_step(grid, x, y, -dx, dy):
            candidates.append((-dx, dy))
        if not walkable(grid, x, y - dy) and can_step(grid, x, y, dx, -dy):
            candidates.append((dx, -dy))

    elif dx != 0:
        if can_step(grid, x, y, dx, 0):
            candidates.append((dx, 0))
        if not walkable(grid, x, y + 1) and can_step(grid, x, y, dx, 1):
            candidates.append((dx, 1))
        if not walkable(grid, x, y - 1) and can_step(grid, x, y, dx, -1):
            candidates.append((dx, -1))

    else:
        if can_step(grid, x, y, 0, dy):
            candidates.append((0, dy))
        if not walkable(grid, x + 1, y) and can_step(grid, x, y, 1, dy):
            candidates.append((1, dy))
        if not walkable(grid, x - 1, y) and can_step(grid, x, y, -1, dy):
            candidates.append((-1, dy))

    # Preserve order but remove accidental duplicates.
    return list(dict.fromkeys(candidates))


def build_jump_function(grid: np.ndarray, goal: Point):
    counter = {'calls': 0}

    @lru_cache(maxsize=None)
    def jump(x: int, y: int, dx: int, dy: int) -> Optional[Point]:
        counter['calls'] += 1
        nx, ny = x + dx, y + dy
        if not can_step(grid, x, y, dx, dy):
            return None
        if (nx, ny) == goal:
            return nx, ny
        if has_forced_neighbour(grid, nx, ny, dx, dy):
            return nx, ny
        if dx != 0 and dy != 0:
            if jump(nx, ny, dx, 0) is not None or jump(nx, ny, 0, dy) is not None:
                return nx, ny
        return jump(nx, ny, dx, dy)

    return jump, counter


# ## 6. Standard JPS, tie-break Smooth JPS, and Smoothness-Aware JPS
# 

# In[9]:


def reconstruct_jps_state_path(
    parent_state: Dict[Tuple[Point, int], Tuple[Point, int]],
    goal_state: Tuple[Point, int],
) -> List[Point]:
    states = [goal_state]
    while states[-1] in parent_state:
        states.append(parent_state[states[-1]])
    states.reverse()
    points = [state[0] for state in states]
    return expand_jump_path(points)


def run_jps(
    grid: np.ndarray,
    start: Point,
    goal: Point,
    *,
    algorithm_name: str = 'Standard JPS',
    heuristic_weight: float = 1.0,
    smooth_tie_break: bool = False,
) -> SearchResult:
    started = time.perf_counter()
    jump, jump_counter = build_jump_function(grid, goal)

    start_state = (start, START_DIRECTION_ID)
    # Scores are keyed by state because incoming direction affects pruning.
    g_score: Dict[Tuple[Point, int], float] = {start_state: 0.0}
    turns_score: Dict[Tuple[Point, int], int] = {start_state: 0}
    parent_state: Dict[Tuple[Point, int], Tuple[Point, int]] = {}

    # (weighted f, turns tie-break, h, g, actual turns, insertion counter, state)
    queue: List[Tuple[float, int, float, float, int, int, Tuple[Point, int]]] = []
    insertion = 0
    h0 = octile_distance(start, goal)
    heapq.heappush(queue, (heuristic_weight * h0, 0, h0, 0.0, 0, insertion, start_state))

    closed_best: Dict[Tuple[Point, int], Tuple[float, int]] = {}
    expanded = generated = pushes = 0

    while queue:
        _, _, _, queued_g, queued_turns, _, state = heapq.heappop(queue)
        current, incoming_id = state
        best_g = g_score.get(state, math.inf)
        best_turns = turns_score.get(state, 10**18)
        if not math.isclose(queued_g, best_g, abs_tol=1e-12) or queued_turns != best_turns:
            continue

        closed_value = closed_best.get(state)
        if closed_value is not None and (
            queued_g > closed_value[0] + 1e-12
            or (math.isclose(queued_g, closed_value[0], abs_tol=1e-12) and queued_turns >= closed_value[1])
        ):
            continue
        closed_best[state] = (queued_g, queued_turns)
        expanded += 1

        if current == goal:
            runtime = (time.perf_counter() - started) * 1000.0
            return SearchResult(
                algorithm_name, True,
                reconstruct_jps_state_path(parent_state, state),
                runtime, expanded, generated, pushes,
                jump_calls=jump_counter['calls'],
                note=f'w={heuristic_weight:g}; smooth={smooth_tie_break}',
            )

        parent_point = None
        if state in parent_state:
            parent_point = parent_state[state][0]

        for dx, dy in pruned_directions(grid, current, parent_point):
            successor = jump(current[0], current[1], dx, dy)
            if successor is None:
                continue
            generated += 1

            successor_direction = movement_direction(current, successor)
            successor_id = DIRECTION_TO_ID[successor_direction]
            successor_state = (successor, successor_id)
            segment_cost = octile_distance(current, successor)
            tentative_g = best_g + segment_cost

            turn_increment = int(incoming_id != START_DIRECTION_ID and incoming_id != successor_id)
            tentative_turns = best_turns + turn_increment

            old_g = g_score.get(successor_state, math.inf)
            old_turns = turns_score.get(successor_state, 10**18)
            distance_better = tentative_g < old_g - 1e-12
            equal_distance_smoother = (
                smooth_tie_break
                and math.isclose(tentative_g, old_g, abs_tol=1e-12)
                and tentative_turns < old_turns
            )

            if not (distance_better or equal_distance_smoother):
                continue

            g_score[successor_state] = tentative_g
            turns_score[successor_state] = tentative_turns
            parent_state[successor_state] = state
            h_value = octile_distance(successor, goal)
            insertion += 1
            heapq.heappush(queue, (
                tentative_g + heuristic_weight * h_value,
                tentative_turns if smooth_tie_break else 0,
                h_value,
                tentative_g,
                tentative_turns,
                insertion,
                successor_state,
            ))
            pushes += 1

    runtime = (time.perf_counter() - started) * 1000.0
    return SearchResult(algorithm_name, False, [], runtime, expanded, generated, pushes,
                        jump_calls=jump_counter['calls'], note='No path')


def direction_angle_degrees(direction_a: Direction, direction_b: Direction) -> float:
    """Return the absolute heading change between two 8-connected directions."""
    if direction_a == direction_b:
        return 0.0
    ax, ay = direction_a
    bx, by = direction_b
    dot = ax * bx + ay * by
    norm = math.hypot(ax, ay) * math.hypot(bx, by)
    cosine = max(-1.0, min(1.0, dot / norm))
    return math.degrees(math.acos(cosine))


def turn_severity_45(incoming_id: int, outgoing_id: int) -> float:
    """Turn severity measured in 45-degree units: 0, 1, 2, 3, or 4."""
    if incoming_id == START_DIRECTION_ID:
        return 0.0
    angle = direction_angle_degrees(DIRECTIONS_8[incoming_id], DIRECTIONS_8[outgoing_id])
    return angle / 45.0


def reconstruct_sa_jps_path(
    parent_state: Dict[Tuple[Point, int], Tuple[Point, int]],
    goal_state: Tuple[Point, int],
) -> List[Point]:
    states = [goal_state]
    while states[-1] in parent_state:
        states.append(parent_state[states[-1]])
    states.reverse()
    return expand_jump_path([state[0] for state in states])


def run_sa_jps(
    grid: np.ndarray,
    start: Point,
    goal: Point,
    *,
    algorithm_name: str = 'SA-JPS',
    turn_weight: float = 0.35,
    heuristic_weight: float = 1.0,
) -> SearchResult:
    """Smoothness-Aware Jump Point Search.

    State
    -----
    (jump_point, incoming_direction)

    Search objective
    ----------------
    objective = geometric_distance + turn_weight * cumulative_turn_severity

    Turn severity is expressed in 45-degree units. Thus, with turn_weight=0.35,
    a 90-degree turn adds 0.70 grid-distance units to the optimization objective.

    heuristic_weight=1.0 gives A*-style ordering for the composite objective.
    heuristic_weight>1.0 creates a faster weighted variant that can sacrifice
    composite-objective optimality and geometric path optimality.
    """
    if turn_weight < 0:
        raise ValueError('turn_weight must be non-negative')
    if heuristic_weight <= 0:
        raise ValueError('heuristic_weight must be positive')

    started = time.perf_counter()
    jump, jump_counter = build_jump_function(grid, goal)

    State = Tuple[Point, int]
    start_state: State = (start, START_DIRECTION_ID)

    distance_score: Dict[State, float] = {start_state: 0.0}
    severity_score: Dict[State, float] = {start_state: 0.0}
    objective_score: Dict[State, float] = {start_state: 0.0}
    parent_state: Dict[State, State] = {}

    # Queue entry:
    # (weighted objective estimate, h_distance, objective_g, distance_g,
    #  severity_g, insertion_order, state)
    queue: List[Tuple[float, float, float, float, float, int, State]] = []
    insertion = 0
    h0 = octile_distance(start, goal)
    heapq.heappush(queue, (
        heuristic_weight * h0, h0, 0.0, 0.0, 0.0, insertion, start_state
    ))

    closed_best: Dict[State, float] = {}
    expanded = generated = pushes = 0

    while queue:
        _, _, queued_objective, queued_distance, queued_severity, _, state = heapq.heappop(queue)

        best_objective = objective_score.get(state, math.inf)
        best_distance = distance_score.get(state, math.inf)
        best_severity = severity_score.get(state, math.inf)
        if not math.isclose(queued_objective, best_objective, abs_tol=1e-12):
            continue
        if not math.isclose(queued_distance, best_distance, abs_tol=1e-12):
            continue
        if not math.isclose(queued_severity, best_severity, abs_tol=1e-12):
            continue

        previous_closed = closed_best.get(state)
        if previous_closed is not None and queued_objective >= previous_closed - 1e-12:
            continue
        closed_best[state] = queued_objective
        expanded += 1

        current, incoming_id = state
        if current == goal:
            runtime = (time.perf_counter() - started) * 1000.0
            result = SearchResult(
                algorithm_name,
                True,
                reconstruct_sa_jps_path(parent_state, state),
                runtime,
                expanded,
                generated,
                pushes,
                jump_calls=jump_counter['calls'],
                note=(
                    f'turn_weight={turn_weight:g}; heuristic_weight={heuristic_weight:g}; '
                    f'objective={queued_objective:.6f}; turn_severity={queued_severity:.3f}'
                ),
            )
            # Dynamic attributes are also copied into benchmark rows by the runner.
            result.search_objective = queued_objective
            result.turn_severity = queued_severity
            result.turn_weight = turn_weight
            result.heuristic_weight = heuristic_weight
            return result

        parent_point = parent_state[state][0] if state in parent_state else None

        for dx, dy in pruned_directions(grid, current, parent_point):
            successor = jump(current[0], current[1], dx, dy)
            if successor is None:
                continue
            generated += 1

            outgoing_direction = movement_direction(current, successor)
            outgoing_id = DIRECTION_TO_ID[outgoing_direction]
            successor_state: State = (successor, outgoing_id)

            segment_cost = octile_distance(current, successor)
            severity_increment = turn_severity_45(incoming_id, outgoing_id)
            tentative_distance = best_distance + segment_cost
            tentative_severity = best_severity + severity_increment
            tentative_objective = tentative_distance + turn_weight * tentative_severity

            old_objective = objective_score.get(successor_state, math.inf)
            old_distance = distance_score.get(successor_state, math.inf)

            objective_better = tentative_objective < old_objective - 1e-12
            exact_objective_shorter = (
                math.isclose(tentative_objective, old_objective, abs_tol=1e-12)
                and tentative_distance < old_distance - 1e-12
            )
            if not (objective_better or exact_objective_shorter):
                continue

            distance_score[successor_state] = tentative_distance
            severity_score[successor_state] = tentative_severity
            objective_score[successor_state] = tentative_objective
            parent_state[successor_state] = state

            h_distance = octile_distance(successor, goal)
            insertion += 1
            heapq.heappush(queue, (
                tentative_objective + heuristic_weight * h_distance,
                h_distance,
                tentative_objective,
                tentative_distance,
                tentative_severity,
                insertion,
                successor_state,
            ))
            pushes += 1

    runtime = (time.perf_counter() - started) * 1000.0
    result = SearchResult(
        algorithm_name, False, [], runtime, expanded, generated, pushes,
        jump_calls=jump_counter['calls'], note='No path'
    )
    result.search_objective = math.nan
    result.turn_severity = math.nan
    result.turn_weight = turn_weight
    result.heuristic_weight = heuristic_weight
    return result


# ==============================================================
# BIDIRECTIONAL JPS FAMILY
# ==============================================================

BidirectionalState = Tuple[Point, int]


def _reverse_direction_id(direction_id: int) -> int:
    """Reverse an 8-connected direction identifier."""
    if direction_id == START_DIRECTION_ID:
        return START_DIRECTION_ID

    dx, dy = DIRECTIONS_8[direction_id]
    return DIRECTION_TO_ID[(-dx, -dy)]


def _trace_state_points(
    parents: Dict[BidirectionalState, BidirectionalState],
    state: BidirectionalState,
) -> List[Point]:
    """
    Trace from a search root to state.

    For the forward search, the root is start.
    For the backward search, the root is goal.
    """
    states = [state]

    while states[-1] in parents:
        states.append(parents[states[-1]])

    states.reverse()
    return [item[0] for item in states]


def _merge_bidirectional_jump_paths(
    forward_parents: Dict[BidirectionalState, BidirectionalState],
    backward_parents: Dict[BidirectionalState, BidirectionalState],
    forward_state: BidirectionalState,
    backward_state: BidirectionalState,
) -> List[Point]:
    """
    Join start->meeting and meeting->goal jump-point chains.
    """
    forward_points = _trace_state_points(
        forward_parents,
        forward_state,
    )

    # Backward tracing produces goal->meeting.
    backward_root_to_meeting = _trace_state_points(
        backward_parents,
        backward_state,
    )
    meeting_to_goal = list(reversed(backward_root_to_meeting))

    jump_points = (
        forward_points
        + meeting_to_goal[1:]
    )

    return expand_jump_path(jump_points)


def _meeting_turn_severity(
    forward_state: BidirectionalState,
    backward_state: BidirectionalState,
) -> float:
    """
    Return the joining turn in 45-degree units.

    The backward state stores the direction used while searching from the
    goal toward the meeting point. The final start->goal path traverses that
    segment in the opposite direction, so the backward direction is reversed
    before the joining angle is evaluated.
    """
    forward_direction_id = forward_state[1]
    backward_search_direction_id = backward_state[1]

    if (
        forward_direction_id == START_DIRECTION_ID
        or backward_search_direction_id == START_DIRECTION_ID
    ):
        return 0.0

    backward_final_direction_id = _reverse_direction_id(
        backward_search_direction_id
    )

    return turn_severity_45(
        forward_direction_id,
        backward_final_direction_id,
    )


def _valid_queue_minimum(
    queue: List[tuple],
    objective_scores: Dict[BidirectionalState, float],
) -> float:
    """
    Return the smallest non-stale queue priority.

    Queue entries use:
        (priority, heuristic, objective, distance, severity,
         insertion, state)
    """
    while queue:
        entry = queue[0]
        priority = float(entry[0])
        queued_objective = float(entry[2])
        state = entry[6]

        current_objective = objective_scores.get(
            state,
            math.inf,
        )

        if math.isclose(
            queued_objective,
            current_objective,
            abs_tol=1e-12,
        ):
            return priority

        heapq.heappop(queue)

    return math.inf


def run_bidirectional_jps(
    grid: np.ndarray,
    start: Point,
    goal: Point,
    *,
    algorithm_name: str = "Bidirectional Standard JPS",
    turn_weight: float = 0.0,
    heuristic_weight: float = 1.0,
    smoothness_aware: bool = False,
) -> SearchResult:
    """
    Bidirectional Jump Point Search.

    Modes
    -----
    smoothness_aware=False, turn_weight=0:
        Bidirectional Standard JPS.

    smoothness_aware=True, heuristic_weight=1:
        Bidirectional SA-JPS.

    smoothness_aware=True, heuristic_weight>1:
        Bidirectional Fast Weighted SA-JPS.

    Notes
    -----
    Each search maintains direction-aware jump-point states. Candidate
    solutions are evaluated whenever the forward and backward searches have
    reached the same grid point. For SA-JPS modes, the joining turn between
    the two partial paths is included in the composite objective.
    """
    if turn_weight < 0:
        raise ValueError(
            "turn_weight must be non-negative"
        )

    if heuristic_weight <= 0:
        raise ValueError(
            "heuristic_weight must be positive"
        )

    started = time.perf_counter()

    if start == goal:
        result = SearchResult(
            algorithm_name,
            True,
            [start],
            (time.perf_counter() - started) * 1000.0,
            0,
            0,
            0,
            note=(
                f"bidirectional; turn_weight={turn_weight:g}; "
                f"heuristic_weight={heuristic_weight:g}"
            ),
        )
        result.search_objective = 0.0
        result.turn_severity = 0.0
        result.turn_weight = turn_weight
        result.heuristic_weight = heuristic_weight
        result.meeting_point = start
        return result

    # The forward jump function terminates when it sees goal.
    # The backward jump function terminates when it sees start.
    jump_forward, jump_counter_forward = build_jump_function(
        grid,
        goal,
    )
    jump_backward, jump_counter_backward = build_jump_function(
        grid,
        start,
    )

    forward_root: BidirectionalState = (
        start,
        START_DIRECTION_ID,
    )
    backward_root: BidirectionalState = (
        goal,
        START_DIRECTION_ID,
    )

    forward_distance: Dict[BidirectionalState, float] = {
        forward_root: 0.0
    }
    backward_distance: Dict[BidirectionalState, float] = {
        backward_root: 0.0
    }

    forward_severity: Dict[BidirectionalState, float] = {
        forward_root: 0.0
    }
    backward_severity: Dict[BidirectionalState, float] = {
        backward_root: 0.0
    }

    forward_objective: Dict[BidirectionalState, float] = {
        forward_root: 0.0
    }
    backward_objective: Dict[BidirectionalState, float] = {
        backward_root: 0.0
    }

    forward_parents: Dict[
        BidirectionalState,
        BidirectionalState,
    ] = {}
    backward_parents: Dict[
        BidirectionalState,
        BidirectionalState,
    ] = {}

    forward_states_by_point: Dict[
        Point,
        Set[BidirectionalState],
    ] = {
        start: {forward_root}
    }
    backward_states_by_point: Dict[
        Point,
        Set[BidirectionalState],
    ] = {
        goal: {backward_root}
    }

    # Queue entry:
    # (priority, heuristic, objective, distance, severity,
    #  insertion, state)
    forward_queue: List[tuple] = []
    backward_queue: List[tuple] = []

    insertion = 0

    forward_h0 = octile_distance(
        start,
        goal,
    )
    backward_h0 = forward_h0

    heapq.heappush(
        forward_queue,
        (
            heuristic_weight * forward_h0,
            forward_h0,
            0.0,
            0.0,
            0.0,
            insertion,
            forward_root,
        ),
    )

    insertion += 1

    heapq.heappush(
        backward_queue,
        (
            heuristic_weight * backward_h0,
            backward_h0,
            0.0,
            0.0,
            0.0,
            insertion,
            backward_root,
        ),
    )

    forward_closed: Dict[
        BidirectionalState,
        float,
    ] = {}
    backward_closed: Dict[
        BidirectionalState,
        float,
    ] = {}

    expanded = 0
    generated = 0
    pushes = 0

    best_complete_objective = math.inf
    best_complete_distance = math.inf
    best_complete_severity = math.inf

    best_forward_state: Optional[
        BidirectionalState
    ] = None
    best_backward_state: Optional[
        BidirectionalState
    ] = None

    def evaluate_meetings_at(
        point: Point,
    ) -> None:
        nonlocal best_complete_objective
        nonlocal best_complete_distance
        nonlocal best_complete_severity
        nonlocal best_forward_state
        nonlocal best_backward_state

        forward_candidates = forward_states_by_point.get(
            point,
            set(),
        )
        backward_candidates = backward_states_by_point.get(
            point,
            set(),
        )

        if not forward_candidates or not backward_candidates:
            return

        for forward_state in forward_candidates:
            for backward_state in backward_candidates:
                join_severity = (
                    _meeting_turn_severity(
                        forward_state,
                        backward_state,
                    )
                    if smoothness_aware
                    else 0.0
                )

                total_distance = (
                    forward_distance[forward_state]
                    + backward_distance[backward_state]
                )

                total_severity = (
                    forward_severity[forward_state]
                    + backward_severity[backward_state]
                    + join_severity
                )

                total_objective = (
                    total_distance
                    + turn_weight * total_severity
                    if smoothness_aware
                    else total_distance
                )

                better_objective = (
                    total_objective
                    < best_complete_objective - 1e-12
                )
                equal_objective_shorter = (
                    math.isclose(
                        total_objective,
                        best_complete_objective,
                        abs_tol=1e-12,
                    )
                    and total_distance
                    < best_complete_distance - 1e-12
                )
                equal_distance_smoother = (
                    math.isclose(
                        total_objective,
                        best_complete_objective,
                        abs_tol=1e-12,
                    )
                    and math.isclose(
                        total_distance,
                        best_complete_distance,
                        abs_tol=1e-12,
                    )
                    and total_severity
                    < best_complete_severity - 1e-12
                )

                if not (
                    better_objective
                    or equal_objective_shorter
                    or equal_distance_smoother
                ):
                    continue

                best_complete_objective = total_objective
                best_complete_distance = total_distance
                best_complete_severity = total_severity
                best_forward_state = forward_state
                best_backward_state = backward_state

    def expand_one_direction(
        *,
        is_forward: bool,
    ) -> bool:
        nonlocal insertion
        nonlocal expanded
        nonlocal generated
        nonlocal pushes

        if is_forward:
            queue = forward_queue
            distance_score = forward_distance
            severity_score = forward_severity
            objective_score = forward_objective
            parents = forward_parents
            states_by_point = forward_states_by_point
            closed = forward_closed
            jump = jump_forward
            target = goal
        else:
            queue = backward_queue
            distance_score = backward_distance
            severity_score = backward_severity
            objective_score = backward_objective
            parents = backward_parents
            states_by_point = backward_states_by_point
            closed = backward_closed
            jump = jump_backward
            target = start

        while queue:
            (
                _,
                _,
                queued_objective,
                queued_distance,
                queued_severity,
                _,
                state,
            ) = heapq.heappop(queue)

            current_objective = objective_score.get(
                state,
                math.inf,
            )
            current_distance = distance_score.get(
                state,
                math.inf,
            )
            current_severity = severity_score.get(
                state,
                math.inf,
            )

            if not math.isclose(
                queued_objective,
                current_objective,
                abs_tol=1e-12,
            ):
                continue

            if not math.isclose(
                queued_distance,
                current_distance,
                abs_tol=1e-12,
            ):
                continue

            if not math.isclose(
                queued_severity,
                current_severity,
                abs_tol=1e-12,
            ):
                continue

            previous_closed = closed.get(state)

            if (
                previous_closed is not None
                and queued_objective
                >= previous_closed - 1e-12
            ):
                continue

            closed[state] = queued_objective
            expanded += 1

            current, incoming_id = state

            evaluate_meetings_at(current)

            parent_point = (
                parents[state][0]
                if state in parents
                else None
            )

            for dx, dy in pruned_directions(
                grid,
                current,
                parent_point,
            ):
                successor = jump(
                    current[0],
                    current[1],
                    dx,
                    dy,
                )

                if successor is None:
                    continue

                generated += 1

                outgoing_direction = movement_direction(
                    current,
                    successor,
                )
                outgoing_id = DIRECTION_TO_ID[
                    outgoing_direction
                ]
                successor_state: BidirectionalState = (
                    successor,
                    outgoing_id,
                )

                segment_cost = octile_distance(
                    current,
                    successor,
                )

                severity_increment = (
                    turn_severity_45(
                        incoming_id,
                        outgoing_id,
                    )
                    if smoothness_aware
                    else 0.0
                )

                tentative_distance = (
                    current_distance
                    + segment_cost
                )
                tentative_severity = (
                    current_severity
                    + severity_increment
                )
                tentative_objective = (
                    tentative_distance
                    + turn_weight * tentative_severity
                    if smoothness_aware
                    else tentative_distance
                )

                old_objective = objective_score.get(
                    successor_state,
                    math.inf,
                )
                old_distance = distance_score.get(
                    successor_state,
                    math.inf,
                )
                old_severity = severity_score.get(
                    successor_state,
                    math.inf,
                )

                objective_better = (
                    tentative_objective
                    < old_objective - 1e-12
                )
                equal_objective_shorter = (
                    math.isclose(
                        tentative_objective,
                        old_objective,
                        abs_tol=1e-12,
                    )
                    and tentative_distance
                    < old_distance - 1e-12
                )
                equal_distance_smoother = (
                    math.isclose(
                        tentative_objective,
                        old_objective,
                        abs_tol=1e-12,
                    )
                    and math.isclose(
                        tentative_distance,
                        old_distance,
                        abs_tol=1e-12,
                    )
                    and tentative_severity
                    < old_severity - 1e-12
                )

                if not (
                    objective_better
                    or equal_objective_shorter
                    or equal_distance_smoother
                ):
                    continue

                distance_score[successor_state] = (
                    tentative_distance
                )
                severity_score[successor_state] = (
                    tentative_severity
                )
                objective_score[successor_state] = (
                    tentative_objective
                )
                parents[successor_state] = state

                states_by_point.setdefault(
                    successor,
                    set(),
                ).add(successor_state)

                heuristic = octile_distance(
                    successor,
                    target,
                )

                insertion += 1
                heapq.heappush(
                    queue,
                    (
                        tentative_objective
                        + heuristic_weight * heuristic,
                        heuristic,
                        tentative_objective,
                        tentative_distance,
                        tentative_severity,
                        insertion,
                        successor_state,
                    ),
                )
                pushes += 1

                evaluate_meetings_at(successor)

            return True

        return False

    while forward_queue and backward_queue:
        minimum_forward = _valid_queue_minimum(
            forward_queue,
            forward_objective,
        )
        minimum_backward = _valid_queue_minimum(
            backward_queue,
            backward_objective,
        )

        if (
            best_forward_state is not None
            and minimum_forward
            >= best_complete_objective - 1e-12
            and minimum_backward
            >= best_complete_objective - 1e-12
        ):
            break

        # Expand the frontier with the smaller current priority.
        if minimum_forward <= minimum_backward:
            progressed = expand_one_direction(
                is_forward=True,
            )
        else:
            progressed = expand_one_direction(
                is_forward=False,
            )

        if not progressed:
            break

    runtime = (
        time.perf_counter() - started
    ) * 1000.0

    total_jump_calls = (
        jump_counter_forward["calls"]
        + jump_counter_backward["calls"]
    )

    if (
        best_forward_state is None
        or best_backward_state is None
    ):
        result = SearchResult(
            algorithm_name,
            False,
            [],
            runtime,
            expanded,
            generated,
            pushes,
            jump_calls=total_jump_calls,
            note=(
                "No bidirectional meeting; "
                f"turn_weight={turn_weight:g}; "
                f"heuristic_weight={heuristic_weight:g}"
            ),
        )
        result.search_objective = math.nan
        result.turn_severity = math.nan
        result.turn_weight = turn_weight
        result.heuristic_weight = heuristic_weight
        result.meeting_point = None
        return result

    path = _merge_bidirectional_jump_paths(
        forward_parents,
        backward_parents,
        best_forward_state,
        best_backward_state,
    )

    valid, reason = validate_path(
        grid,
        path,
        start,
        goal,
    )

    if not valid:
        result = SearchResult(
            algorithm_name,
            False,
            [],
            runtime,
            expanded,
            generated,
            pushes,
            jump_calls=total_jump_calls,
            note=(
                "Invalid merged path: "
                f"{reason}"
            ),
        )
        result.search_objective = math.nan
        result.turn_severity = math.nan
        result.turn_weight = turn_weight
        result.heuristic_weight = heuristic_weight
        result.meeting_point = best_forward_state[0]
        return result

    result = SearchResult(
        algorithm_name,
        True,
        path,
        runtime,
        expanded,
        generated,
        pushes,
        jump_calls=total_jump_calls,
        note=(
            "bidirectional; "
            f"turn_weight={turn_weight:g}; "
            f"heuristic_weight={heuristic_weight:g}; "
            f"objective={best_complete_objective:.6f}; "
            f"turn_severity={best_complete_severity:.3f}; "
            f"meeting={best_forward_state[0]}"
        ),
    )

    result.search_objective = (
        best_complete_objective
    )
    result.turn_severity = (
        best_complete_severity
    )
    result.turn_weight = turn_weight
    result.heuristic_weight = heuristic_weight
    result.meeting_point = best_forward_state[0]

    return result


def run_bidirectional_standard_jps(
    grid: np.ndarray,
    start: Point,
    goal: Point,
    *,
    algorithm_name: str = "Bidirectional Standard JPS",
    heuristic_weight: float = 1.0,
) -> SearchResult:
    return run_bidirectional_jps(
        grid,
        start,
        goal,
        algorithm_name=algorithm_name,
        turn_weight=0.0,
        heuristic_weight=heuristic_weight,
        smoothness_aware=False,
    )


def run_bidirectional_sa_jps(
    grid: np.ndarray,
    start: Point,
    goal: Point,
    *,
    algorithm_name: str = "Bidirectional SA-JPS",
    turn_weight: float = 0.35,
    heuristic_weight: float = 1.0,
) -> SearchResult:
    return run_bidirectional_jps(
        grid,
        start,
        goal,
        algorithm_name=algorithm_name,
        turn_weight=turn_weight,
        heuristic_weight=heuristic_weight,
        smoothness_aware=True,
    )


# ## 7. Experimental JPS–Dijkstra corridor hybrid
# 
# This hybrid is intentionally different from simply setting the JPS heuristic to zero.
# 
# 1. Standard JPS obtains a feasible path quickly.
# 2. A corridor of configurable radius is formed around that path.
# 3. Dijkstra searches direction-aware states only inside the corridor.
# 4. The objective is lexicographic: minimum distance first, then minimum number of turns.
# 
# Because the original JPS path remains inside the corridor, the refinement stage always has at least that route available. Increasing the radius gives Dijkstra more freedom but increases expansions and runtime.

# In[10]:


def build_corridor_mask(grid: np.ndarray, seed_path: Sequence[Point], radius: int) -> np.ndarray:
    mask = np.zeros_like(grid, dtype=bool)
    height, width = grid.shape
    for x, y in seed_path:
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        mask[y0:y1, x0:x1] = True
    return mask & grid


def reconstruct_direction_state_path(
    parents: Dict[Tuple[Point, int], Tuple[Point, int]],
    goal_state: Tuple[Point, int],
) -> List[Point]:
    states = [goal_state]
    while states[-1] in parents:
        states.append(parents[states[-1]])
    states.reverse()
    return [state[0] for state in states]


def corridor_dijkstra(
    grid: np.ndarray,
    corridor: np.ndarray,
    start: Point,
    goal: Point,
) -> SearchResult:
    started = time.perf_counter()
    start_state = (start, START_DIRECTION_ID)
    best: Dict[Tuple[Point, int], Tuple[float, int]] = {start_state: (0.0, 0)}
    parents: Dict[Tuple[Point, int], Tuple[Point, int]] = {}
    queue: List[Tuple[float, int, int, Tuple[Point, int]]] = [(0.0, 0, 0, start_state)]
    insertion = 0
    expanded = generated = pushes = 0

    while queue:
        distance, turns, _, state = heapq.heappop(queue)
        if best.get(state) != (distance, turns):
            continue
        expanded += 1
        point, incoming_id = state
        if point == goal:
            runtime = (time.perf_counter() - started) * 1000.0
            return SearchResult('Corridor Dijkstra refinement', True,
                                reconstruct_direction_state_path(parents, state), runtime,
                                expanded, generated, pushes)

        x, y = point
        for dx, dy in DIRECTIONS_8:
            nx, ny = x + dx, y + dy
            if not can_step(grid, x, y, dx, dy) or not corridor[ny, nx]:
                continue
            generated += 1
            direction_id = DIRECTION_TO_ID[(dx, dy)]
            state2 = ((nx, ny), direction_id)
            distance2 = distance + (SQRT2 if dx and dy else 1.0)
            turns2 = turns + int(incoming_id != START_DIRECTION_ID and incoming_id != direction_id)
            candidate = (distance2, turns2)
            old = best.get(state2)
            if old is None or candidate[0] < old[0] - 1e-12 or (
                math.isclose(candidate[0], old[0], abs_tol=1e-12) and candidate[1] < old[1]
            ):
                best[state2] = candidate
                parents[state2] = state
                insertion += 1
                heapq.heappush(queue, (distance2, turns2, insertion, state2))
                pushes += 1

    runtime = (time.perf_counter() - started) * 1000.0
    return SearchResult('Corridor Dijkstra refinement', False, [], runtime,
                        expanded, generated, pushes, note='No corridor path')


def run_jps_dijkstra_hybrid(
    grid: np.ndarray,
    start: Point,
    goal: Point,
    corridor_radius: int = 4,
) -> SearchResult:
    total_started = time.perf_counter()
    seed = run_jps(grid, start, goal, algorithm_name='Standard JPS')
    if not seed.success:
        seed.algorithm = 'JPS-Dijkstra Corridor'
        seed.note = 'JPS seed failed'
        return seed

    corridor = build_corridor_mask(grid, seed.path, corridor_radius)
    refined = corridor_dijkstra(grid, corridor, start, goal)
    total_runtime = (time.perf_counter() - total_started) * 1000.0

    if not refined.success:
        return SearchResult(
            'JPS-Dijkstra Corridor', True, seed.path, total_runtime,
            seed.expanded_nodes + refined.expanded_nodes,
            seed.generated_successors + refined.generated_successors,
            seed.queue_pushes + refined.queue_pushes,
            jump_calls=seed.jump_calls,
            note=f'radius={corridor_radius}; refinement failed, seed retained',
        )

    return SearchResult(
        'JPS-Dijkstra Corridor', True, refined.path, total_runtime,
        seed.expanded_nodes + refined.expanded_nodes,
        seed.generated_successors + refined.generated_successors,
        seed.queue_pushes + refined.queue_pushes,
        jump_calls=seed.jump_calls,
        note=f'radius={corridor_radius}',
    )
def run_fast_sa_jps_dijkstra_hybrid(
    grid: np.ndarray,
    start: Point,
    goal: Point,
    *,
    corridor_radius: int = 4,
    turn_weight: float = 0.35,
    heuristic_weight: float = 1.15,
    algorithm_name: str = "Fast Weighted SA-JPS-Dijkstra Corridor",
) -> SearchResult:
    """
    Two-stage hybrid planner.

    Stage 1
    -------
    Fast Weighted SA-JPS creates a smoothness-aware, goal-directed seed path.

    Stage 2
    -------
    Dijkstra searches only inside a corridor constructed around the seed path.
    The refinement minimizes path length and uses turn count as a secondary
    tie-break criterion.

    If corridor refinement fails, the valid Fast Weighted SA-JPS seed path is
    retained.
    """
    if corridor_radius < 0:
        raise ValueError("corridor_radius must be non-negative")
    if turn_weight < 0:
        raise ValueError("turn_weight must be non-negative")
    if heuristic_weight <= 0:
        raise ValueError("heuristic_weight must be positive")

    total_started = time.perf_counter()

    seed = run_sa_jps(
        grid,
        start,
        goal,
        algorithm_name="Fast Weighted SA-JPS seed",
        turn_weight=float(turn_weight),
        heuristic_weight=float(heuristic_weight),
    )

    if not seed.success:
        seed.algorithm = algorithm_name
        seed.note = (
            "Fast Weighted SA-JPS seed failed; "
            f"radius={corridor_radius}; "
            f"turn_weight={turn_weight:g}; "
            f"heuristic_weight={heuristic_weight:g}"
        )
        seed.search_objective = getattr(seed, "search_objective", np.nan)
        seed.turn_severity = getattr(seed, "turn_severity", np.nan)
        seed.turn_weight = float(turn_weight)
        seed.heuristic_weight = float(heuristic_weight)
        seed.corridor_radius = int(corridor_radius)
        seed.seed_path_length = np.nan
        seed.refinement_used = False
        return seed

    corridor = build_corridor_mask(
        grid,
        seed.path,
        int(corridor_radius),
    )

    # Explicitly retain endpoints even for radius=0.
    corridor[start[1], start[0]] = True
    corridor[goal[1], goal[0]] = True

    refined = corridor_dijkstra(
        grid,
        corridor,
        start,
        goal,
    )

    total_runtime = (
        time.perf_counter() - total_started
    ) * 1000.0

    seed_length = path_length(seed.path)
    seed_turns, seed_angle, _ = path_turn_metrics(seed.path)

    if refined.success:
        final_path = refined.path
        refinement_used = True
        note = (
            f"radius={corridor_radius}; "
            f"turn_weight={turn_weight:g}; "
            f"heuristic_weight={heuristic_weight:g}; "
            f"seed_length={seed_length:.6f}; "
            f"seed_turns={seed_turns}; "
            f"seed_turning_angle={seed_angle:.3f}; "
            "corridor_refinement=success"
        )
    else:
        final_path = seed.path
        refinement_used = False
        note = (
            f"radius={corridor_radius}; "
            f"turn_weight={turn_weight:g}; "
            f"heuristic_weight={heuristic_weight:g}; "
            f"seed_length={seed_length:.6f}; "
            f"seed_turns={seed_turns}; "
            f"seed_turning_angle={seed_angle:.3f}; "
            "corridor_refinement=failed; seed retained"
        )

    result = SearchResult(
        algorithm_name,
        True,
        final_path,
        total_runtime,
        seed.expanded_nodes + refined.expanded_nodes,
        seed.generated_successors + refined.generated_successors,
        seed.queue_pushes + refined.queue_pushes,
        jump_calls=seed.jump_calls,
        note=note,
    )

    # Preserve research-relevant parameters and stage information.
    result.search_objective = getattr(
        seed,
        "search_objective",
        np.nan,
    )
    result.turn_severity = getattr(
        seed,
        "turn_severity",
        np.nan,
    )
    result.turn_weight = float(turn_weight)
    result.heuristic_weight = float(heuristic_weight)
    result.corridor_radius = int(corridor_radius)
    result.seed_path_length = float(seed_length)
    result.seed_turns = int(seed_turns)
    result.seed_turning_angle_deg = float(seed_angle)
    result.refinement_used = bool(refinement_used)
    result.corridor_cells = int(corridor.sum())

    return result


# ## 8. Unified algorithm runner and validity checks

# In[11]:


ALGORITHM_ORDER = [
    'A*',
    'Standard JPS',
    'Bidirectional Standard JPS',
    'Smooth JPS',
    'SA-JPS',
    'Bidirectional SA-JPS',
    'Fast Weighted SA-JPS',
    'Bidirectional Fast Weighted SA-JPS',
    'JPS-Dijkstra Corridor',
    'Fast Weighted SA-JPS-Dijkstra Corridor',
]


def validate_path(
    grid: np.ndarray,
    path: Sequence[Point],
    start: Point,
    goal: Point,
) -> Tuple[bool, str]:
    if not path:
        return False, 'Empty path'
    if path[0] != start or path[-1] != goal:
        return False, 'Incorrect endpoints'
    for point in path:
        if not walkable(grid, *point):
            return False, f'Blocked point {point}'
    for a, b in zip(path[:-1], path[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        if abs(dx) > 1 or abs(dy) > 1 or (dx == 0 and dy == 0):
            return False, f'Non-adjacent movement {a}->{b}'
        if not can_step(grid, a[0], a[1], dx, dy):
            return False, f'Illegal movement {a}->{b}'
    return True, ''


def normalise_approaches(
    approaches: Optional[Sequence[str]] = None,
) -> List[str]:
    """Validate and normalise a requested algorithm subset."""
    if approaches is None:
        return list(ALGORITHM_ORDER)

    if any(isinstance(item, dict) for item in approaches):
        # run_all_algorithms already accepts labeled configurations. Preserve
        # them here instead of converting their dictionaries to strings.
        selected = []
        labels = set()
        for item in approaches:
            algorithm = str(item.get("algorithm", "")) if isinstance(item, dict) else str(item)
            label = str(item.get("label", algorithm)) if isinstance(item, dict) else algorithm
            if algorithm not in ALGORITHM_ORDER:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            if not label or label in labels:
                raise ValueError(f"Empty or duplicate configuration label: {label}")
            labels.add(label)
            selected.append(item)
        for label in labels:
            if label not in ALGORITHM_ORDER:
                ALGORITHM_ORDER.append(label)
        return selected

    requested = [str(name).strip() for name in approaches if str(name).strip()]
    unknown = [name for name in requested if name not in ALGORITHM_ORDER]
    if unknown:
        raise ValueError(
            f"Unknown approaches: {unknown}. "
            f"Available approaches are: {ALGORITHM_ORDER}"
        )

    # Preserve canonical order and remove duplicates.
    return [name for name in ALGORITHM_ORDER if name in set(requested)]


# ## 9. Synthetic smoke test
# 
# Run this before the full dataset. Every successful method should produce a valid route.

# In[12]:


def run_synthetic_smoke_test() -> pd.DataFrame:
    """Run every configured algorithm on a small artificial map."""
    synthetic = np.ones((30, 40), dtype=bool)
    synthetic[5:25, 18] = False
    synthetic[14, 18] = True
    synthetic[4:20, 28] = False
    synthetic[9, 28] = True

    synthetic_start = (2, 2)
    synthetic_goal = (37, 27)

    smoke_results = run_all_algorithms(
        synthetic,
        synthetic_start,
        synthetic_goal,
    )
    smoke_table = pd.DataFrame(
        [result.metrics() for result in smoke_results]
    )
    columns = [
        'Algorithm', 'Success', 'Path length', 'Turns',
        'Expanded nodes', 'Time ms', 'Note',
    ]
    return smoke_table[columns]


# ## 10. Select maps and scenarios

# In[13]:


def parse_map_filename(filename: str) -> Optional[Tuple[str, int, int]]:
    match = re.match(r'(.+)_([0-9]+)_([0-9]+)\.map$', Path(filename).name)
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def discover_selected_maps_ds1() -> List[Path]:
    selected = []
    for path in MAPS_DIR.rglob('*.map'):
        parsed = parse_map_filename(path.name)
        if parsed is None:
            continue
        _, variant, size = parsed
        if variant in CONFIG['map_variant'] and size in CONFIG['map_sizes']:
            selected.append(path)
    selected.sort(key=lambda p: p.name)
    limit = CONFIG['map_limit']
    return selected if limit is None else selected[:int(limit)]


def discover_selected_maps() -> List[Path]:
    selected = []
#     print(list(MAPS_DIR.rglob('*.map')))
    for path in MAPS_DIR.rglob('*.map'):
        parsed = parse_map_filename(path.name)
#         if parsed is None:
#             continue
#         _, variant, size = parsed
#         if variant in CONFIG['map_variant'] and size in CONFIG['map_sizes']:
        selected.append(path)
#         print(path)
    selected.sort(key=lambda p: p.name)
    limit = CONFIG['map_limit']
    return selected if limit is None else selected[:int(limit)]



def select_scenarios(
    frame: pd.DataFrame,
    count: Optional[int],
    mode: str,
) -> pd.DataFrame:
    """
    Select scenarios for one map.

    count=None uses every valid scenario in the scenario file.
    """
    frame = frame.dropna(
        subset=['start_x', 'start_y', 'goal_x', 'goal_y']
    ).copy()

    # Preserve the original row number so a scenario can be reproduced later.
    frame['source_scenario_index'] = frame.index.astype(int)

    if count is None:
        return frame.reset_index(drop=True)

    count = max(0, min(int(count), len(frame)))
    if count == 0:
        return frame.iloc[0:0].reset_index(drop=True)

    mode = str(mode).strip().lower()

    if mode == 'longest':
        selected = frame.nlargest(count, 'reference_length')
    elif mode == 'shortest':
        selected = frame.nsmallest(count, 'reference_length')
    elif mode == 'uniform':
        ordered = frame.sort_values('reference_length')
        indices = np.linspace(0, len(ordered) - 1, count, dtype=int)
        selected = ordered.iloc[indices]
    else:
        raise ValueError(
            "scenario_selection must be 'longest', 'shortest', or 'uniform'"
        )

    return selected.reset_index(drop=True)


# In[14]:


# ==============================================================
# MAP STRUCTURE ANALYSIS AND MAP-TYPE CLASSIFICATION
# ==============================================================

from collections import deque


MAP_FEATURE_COLUMNS = [
    "Obstacle density",
    "Free-cell ratio",
    "Dead-end ratio",
    "Corridor ratio",
    "Junction ratio",
    "Obstacle-edge density",
    "Mean axis visibility",
    "Connected components",
    "Largest component ratio",
]


def _map_free_mask(grid: np.ndarray) -> np.ndarray:
    """Return a Boolean mask where True denotes a traversable cell."""
    return np.asarray(grid) == 0


def _cardinal_neighbor_count(free: np.ndarray) -> np.ndarray:
    """Count traversable four-connected neighbours for every grid cell."""
    padded = np.pad(
        free.astype(np.uint8),
        1,
        mode="constant",
        constant_values=0,
    )
    return (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )


def _connected_component_features(
    free: np.ndarray,
) -> Tuple[int, float]:
    """Return free-space component count and largest-component ratio."""
    height, width = free.shape
    visited = np.zeros_like(free, dtype=bool)
    component_sizes: List[int] = []

    for y in range(height):
        for x in range(width):
            if not free[y, x] or visited[y, x]:
                continue

            queue = deque([(x, y)])
            visited[y, x] = True
            size = 0

            while queue:
                current_x, current_y = queue.popleft()
                size += 1

                for dx, dy in (
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                ):
                    next_x = current_x + dx
                    next_y = current_y + dy

                    if (
                        0 <= next_x < width
                        and 0 <= next_y < height
                        and free[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        queue.append((next_x, next_y))

            component_sizes.append(size)

    free_count = int(free.sum())
    if free_count == 0 or not component_sizes:
        return 0, 0.0

    return (
        len(component_sizes),
        float(max(component_sizes) / free_count),
    )


def _obstacle_edge_density(free: np.ndarray) -> float:
    """Fraction of adjacent cell pairs that cross a free/obstacle boundary."""
    horizontal_changes = int(
        (free[:, 1:] != free[:, :-1]).sum()
    )
    vertical_changes = int(
        (free[1:, :] != free[:-1, :]).sum()
    )
    possible_edges = (
        free.shape[0] * max(0, free.shape[1] - 1)
        + max(0, free.shape[0] - 1) * free.shape[1]
    )
    if possible_edges == 0:
        return 0.0
    return float(
        (horizontal_changes + vertical_changes)
        / possible_edges
    )


def _mean_axis_visibility(
    free: np.ndarray,
    *,
    sample_count: int = 1000,
    max_distance: int = 40,
    random_seed: int = 42,
) -> float:
    """
    Estimate normalized cardinal-axis visibility.

    A value near zero indicates locally enclosed free space; a value near one
    indicates long unobstructed cardinal sight lines.
    """
    free_points = np.argwhere(free)
    if len(free_points) == 0:
        return 0.0

    rng = np.random.default_rng(random_seed)
    number_to_sample = min(int(sample_count), len(free_points))
    selected = rng.choice(
        len(free_points),
        size=number_to_sample,
        replace=False,
    )

    height, width = free.shape
    values: List[float] = []

    for index in selected:
        y, x = free_points[index]
        directional: List[float] = []

        for dx, dy in (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ):
            distance = 0
            for step in range(1, int(max_distance) + 1):
                next_x = int(x + dx * step)
                next_y = int(y + dy * step)

                if (
                    next_x < 0
                    or next_x >= width
                    or next_y < 0
                    or next_y >= height
                    or not free[next_y, next_x]
                ):
                    break
                distance += 1

            directional.append(
                distance / max(1, int(max_distance))
            )

        values.append(float(np.mean(directional)))

    return float(np.mean(values))


def analyze_map_structure(
    grid: np.ndarray,
    *,
    visibility_samples: int = 1000,
    visibility_distance: int = 40,
    random_seed: int = 42,
) -> Dict[str, float]:
    """Calculate interpretable structural descriptors for an occupancy grid."""
    free = _map_free_mask(grid)
    total_cells = int(free.size)
    free_cells = int(free.sum())
    obstacle_cells = total_cells - free_cells

    if total_cells == 0:
        raise ValueError("Cannot analyse an empty grid.")

    if free_cells == 0:
        return {
            "Map width": int(grid.shape[1]),
            "Map height": int(grid.shape[0]),
            "Free-cell ratio": 0.0,
            "Obstacle density": 1.0,
            "Dead-end ratio": 0.0,
            "Corridor ratio": 0.0,
            "Junction ratio": 0.0,
            "Obstacle-edge density": 0.0,
            "Mean axis visibility": 0.0,
            "Connected components": 0,
            "Largest component ratio": 0.0,
        }

    neighbour_count = _cardinal_neighbor_count(free)
    free_neighbour_counts = neighbour_count[free]

    dead_end_ratio = float(
        np.mean(free_neighbour_counts <= 1)
    )
    junction_ratio = float(
        np.mean(free_neighbour_counts >= 3)
    )

    left = np.zeros_like(free)
    right = np.zeros_like(free)
    up = np.zeros_like(free)
    down = np.zeros_like(free)

    left[:, 1:] = free[:, :-1]
    right[:, :-1] = free[:, 1:]
    up[1:, :] = free[:-1, :]
    down[:-1, :] = free[1:, :]

    horizontal_corridor = left & right & ~up & ~down
    vertical_corridor = up & down & ~left & ~right
    corridor_cells = horizontal_corridor | vertical_corridor
    corridor_ratio = float(corridor_cells[free].mean())

    component_count, largest_component_ratio = (
        _connected_component_features(free)
    )

    return {
        "Map width": int(grid.shape[1]),
        "Map height": int(grid.shape[0]),
        "Free-cell ratio": float(free_cells / total_cells),
        "Obstacle density": float(obstacle_cells / total_cells),
        "Dead-end ratio": dead_end_ratio,
        "Corridor ratio": corridor_ratio,
        "Junction ratio": junction_ratio,
        "Obstacle-edge density": _obstacle_edge_density(free),
        "Mean axis visibility": _mean_axis_visibility(
            free,
            sample_count=visibility_samples,
            max_distance=visibility_distance,
            random_seed=random_seed,
        ),
        "Connected components": int(component_count),
        "Largest component ratio": largest_component_ratio,
    }


def classify_map_type(
    features: Dict[str, float],
) -> str:
    """
    Assign an interpretable structural map type.

    These transparent rules are intended for reporting and should be
    interpreted together with the continuous descriptors. Thresholds can be
    adjusted in CONFIG after visual validation on representative maps.
    """
    density = float(features["Obstacle density"])
    visibility = float(features["Mean axis visibility"])
    corridor = float(features["Corridor ratio"])
    dead_end = float(features["Dead-end ratio"])
    junction = float(features["Junction ratio"])
    edge_density = float(features["Obstacle-edge density"])
    largest_component = float(features["Largest component ratio"])

    open_density_max = float(
        CONFIG.get("map_type_open_density_max", 0.18)
    )
    open_visibility_min = float(
        CONFIG.get("map_type_open_visibility_min", 0.45)
    )
    open_corridor_max = float(
        CONFIG.get("map_type_open_corridor_max", 0.10)
    )

    maze_corridor_min = float(
        CONFIG.get("map_type_maze_corridor_min", 0.22)
    )
    maze_visibility_max = float(
        CONFIG.get("map_type_maze_visibility_max", 0.16)
    )
    maze_dead_end_min = float(
        CONFIG.get("map_type_maze_dead_end_min", 0.025)
    )

    urban_edge_min = float(
        CONFIG.get("map_type_urban_edge_min", 0.10)
    )
    urban_junction_min = float(
        CONFIG.get("map_type_urban_junction_min", 0.15)
    )

    if (
        density <= open_density_max
        and visibility >= open_visibility_min
        and corridor <= open_corridor_max
    ):
        return "Open"

    if (
        corridor >= maze_corridor_min
        or (
            visibility <= maze_visibility_max
            and dead_end >= maze_dead_end_min
        )
    ):
        return "Maze"

    if (
        0.15 <= density <= 0.55
        and edge_density >= urban_edge_min
        and junction >= urban_junction_min
        and largest_component >= 0.80
    ):
        return "Urban"

    return "Mixed"


def get_map_type(
    grid: np.ndarray,
) -> Tuple[str, Dict[str, float]]:
    """Return rule-based map type and all structural descriptors."""
    features = analyze_map_structure(
        grid,
        visibility_samples=int(
            CONFIG.get("map_visibility_samples", 1000)
        ),
        visibility_distance=int(
            CONFIG.get("map_visibility_distance", 40)
        ),
        random_seed=int(
            CONFIG.get("random_seed", 42)
        ),
    )
    return classify_map_type(features), features


def build_map_feature_table(
    map_paths: Optional[Sequence[Path]] = None,
) -> pd.DataFrame:
    """Analyse each selected map exactly once and return one row per map."""
    paths = list(selected_maps if map_paths is None else map_paths)
    rows: List[Dict[str, object]] = []

    for map_path in tqdm(
        paths,
        desc="Analysing map structure",
        unit="map",
    ):
        grid = read_map(map_path)
        map_type, features = get_map_type(grid)
        rows.append(
            {
                "Map name": map_path.name,
                "Map type": map_type,
                **features,
            }
        )

    return pd.DataFrame(rows)


def add_structural_clusters(
    map_feature_df: pd.DataFrame,
    *,
    number_of_clusters: int = 4,
) -> pd.DataFrame:
    """
    Add optional data-driven structural clusters.

    Cluster identifiers are deliberately neutral. Inspect the saved cluster
    profiles before assigning descriptive names such as open or maze-like.
    """
    output = map_feature_df.copy()

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        output["Structural cluster"] = np.nan
        print(
            "scikit-learn is unavailable; structural clustering was skipped."
        )
        return output

    usable_columns = [
        column
        for column in MAP_FEATURE_COLUMNS
        if column in output.columns
        and column != "Connected components"
    ]

    if len(output) < 2 or not usable_columns:
        output["Structural cluster"] = np.nan
        return output

    cluster_count = max(
        2,
        min(
            int(number_of_clusters),
            len(output),
        ),
    )

    X = (
        output[usable_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    X_scaled = StandardScaler().fit_transform(X)

    model = KMeans(
        n_clusters=cluster_count,
        random_state=int(CONFIG.get("random_seed", 42)),
        n_init=20,
    )
    output["Structural cluster"] = model.fit_predict(X_scaled)
    return output


# ## 11. Dataset benchmark
# 
# Runtime includes only the algorithm itself. File reading, result construction, and plotting are outside the timed section.

# In[15]:


BENCHMARK_CACHE_VERSION = 5


def _benchmark_cache_signature(
    approaches: Sequence[str],
) -> Dict[str, object]:
    """
    Build a deterministic signature for the expensive benchmark.

    The cache is invalidated when maps, scenarios, selected algorithms,
    relevant parameters, or the cache schema version change.
    """
    map_records = []
    for map_path in selected_maps:
        scenario_path = locate_file(
            SCENARIOS_DIR,
            map_path.name + '.scen',
        )
        map_records.append({
            'map': str(map_path.resolve()),
            'map_size': map_path.stat().st_size,
            'map_mtime_ns': map_path.stat().st_mtime_ns,
            'scenario': str(scenario_path.resolve()),
            'scenario_size': scenario_path.stat().st_size,
            'scenario_mtime_ns': scenario_path.stat().st_mtime_ns,
        })

    return {
        'cache_version': BENCHMARK_CACHE_VERSION,
        'dataset_name': CONFIG.get('dataset_name'),
        'maps': map_records,
        'approaches': list(approaches),
        'scenarios_per_map': CONFIG.get('scenarios_per_map'),
        'scenario_selection': CONFIG.get('scenario_selection'),
        'random_seed': CONFIG.get('random_seed'),
        'sa_turn_weight': CONFIG.get('sa_turn_weight'),
        'fast_sa_heuristic_weight': CONFIG.get(
            'fast_sa_heuristic_weight'
        ),
        'hybrid_corridor_radius': CONFIG.get(
            'hybrid_corridor_radius'
        ),
    }


def _benchmark_cache_paths(
    signature: Dict[str, object],
) -> Tuple[Path, Path]:
    payload = json.dumps(
        signature,
        sort_keys=True,
        default=str,
    ).encode('utf-8')
    digest = hashlib.sha256(payload).hexdigest()[:16]

    cache_dir = Path(
        CONFIG.get('benchmark_cache_dir', RESULTS_DIR / 'cache')
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    return (
        dataset_output_path(cache_dir / f'benchmark_{digest}.pkl'),
        dataset_output_path(cache_dir / f'benchmark_{digest}.json'),
    )


def _load_cached_benchmark(
    data_path: Path,
    metadata_path: Path,
    expected_signature: Dict[str, object],
) -> Optional[pd.DataFrame]:
    if not data_path.exists() or not metadata_path.exists():
        return None

    try:
        with metadata_path.open('r', encoding='utf-8') as f:
            metadata = json.load(f)
        if metadata.get('signature') != expected_signature:
            return None

        frame = pd.read_pickle(data_path)
        if 'Algorithm' in frame.columns:
            frame['Algorithm'] = pd.Categorical(
                frame['Algorithm'].astype(str),
                categories=ALGORITHM_ORDER,
                ordered=True,
            )
        print(f'Loaded benchmark cache: {data_path.resolve()}')
        return frame
    except Exception as exc:
        print(f'Ignoring unreadable benchmark cache: {exc}')
        return None


def _save_benchmark_cache(
    frame: pd.DataFrame,
    data_path: Path,
    metadata_path: Path,
    signature: Dict[str, object],
) -> None:
    _atomic_pickle_save(frame, data_path)
    metadata = {
        'signature': signature,
        'rows': int(len(frame)),
        'columns': list(frame.columns),
    }
    _atomic_json_save(metadata, metadata_path)
#     print(f'Saved benchmark cache: {data_path.resolve()}')

def run_all_algorithms(
    grid: np.ndarray,
    start: Point,
    goal: Point,
    approaches: Optional[Sequence[object]] = None,
) -> List[SearchResult]:
    """
    Run selected algorithms.

    Supported input formats
    -----------------------
    1. Simple algorithm-name list:

        approaches=[
            "A*",
            "Standard JPS",
            "SA-JPS",
        ]

    2. Labeled configuration list:

        approaches=[
            {
                "label": "SA-JPS tw=0.20",
                "algorithm": "SA-JPS",
                "parameters": {
                    "turn_weight": 0.20,
                    "heuristic_weight": 1.00,
                },
            },
            {
                "label": "SA-JPS tw=0.50",
                "algorithm": "SA-JPS",
                "parameters": {
                    "turn_weight": 0.50,
                    "heuristic_weight": 1.00,
                },
            },
        ]

    The labeled form permits the same base algorithm to be executed
    repeatedly with different parameter settings.
    """

    # ----------------------------------------------------------
    # Convert input into a common configuration format
    # ----------------------------------------------------------
    if approaches is None:
        run_configs = [
            {
                "label": algorithm_name,
                "algorithm": algorithm_name,
                "parameters": {},
            }
            for algorithm_name in ALGORITHM_ORDER
        ]

    else:
        run_configs = []

        for item in approaches:
            # Simple string form
            if isinstance(item, str):
                algorithm_name = item.strip()

                if algorithm_name not in ALGORITHM_ORDER:
                    raise ValueError(
                        f"Unknown algorithm {algorithm_name!r}. "
                        f"Available algorithms are: {ALGORITHM_ORDER}"
                    )

                run_configs.append(
                    {
                        "label": algorithm_name,
                        "algorithm": algorithm_name,
                        "parameters": {},
                    }
                )
                continue

            # Labeled dictionary form
            if not isinstance(item, dict):
                raise TypeError(
                    "Every approaches entry must be either an algorithm "
                    "name string or a configuration dictionary."
                )

            algorithm_name = str(
                item.get("algorithm", "")
            ).strip()

            label = str(
                item.get("label", algorithm_name)
            ).strip()

            parameters = item.get(
                "parameters",
                {},
            )

            if algorithm_name not in ALGORITHM_ORDER:
                raise ValueError(
                    f"Unknown algorithm {algorithm_name!r}. "
                    f"Available algorithms are: {ALGORITHM_ORDER}"
                )

            if not label:
                raise ValueError(
                    "Every labeled approach must have a non-empty label."
                )

            if not isinstance(parameters, dict):
                raise TypeError(
                    f"Parameters for {label!r} must be a dictionary."
                )

            run_configs.append(
                {
                    "label": label,
                    "algorithm": algorithm_name,
                    "parameters": dict(parameters),
                }
            )

    # Labels must be unique because they are used in plots and CSV rows.
    labels = [
        config["label"]
        for config in run_configs
    ]

    duplicate_labels = sorted(
        {
            label
            for label in labels
            if labels.count(label) > 1
        }
    )

    if duplicate_labels:
        raise ValueError(
            f"Approach labels must be unique. "
            f"Duplicate labels: {duplicate_labels}"
        )

    # ----------------------------------------------------------
    # Execute one configuration
    # ----------------------------------------------------------
    def run_configuration(
        config: Dict[str, object],
    ) -> SearchResult:

        label = str(config["label"])
        algorithm_name = str(config["algorithm"])
        parameters = dict(config["parameters"])

        if algorithm_name == "A*":
            unknown = set(parameters)

            if unknown:
                raise ValueError(
                    f"A* does not accept these parameters: "
                    f"{sorted(unknown)}"
                )

            result = run_astar(
                grid,
                start,
                goal,
            )

        elif algorithm_name == "Standard JPS":
            allowed = {
                "heuristic_weight",
                "smooth_tie_break",
            }

            unknown = set(parameters) - allowed

            if unknown:
                raise ValueError(
                    f"Unsupported Standard JPS parameters: "
                    f"{sorted(unknown)}"
                )

            result = run_jps(
                grid,
                start,
                goal,
                algorithm_name=label,
                heuristic_weight=float(
                    parameters.get(
                        "heuristic_weight",
                        1.0,
                    )
                ),
                smooth_tie_break=bool(
                    parameters.get(
                        "smooth_tie_break",
                        False,
                    )
                ),
            )


        elif algorithm_name == "Bidirectional Standard JPS":
            allowed = {
                "heuristic_weight",
            }

            unknown = set(parameters) - allowed

            if unknown:
                raise ValueError(
                    "Unsupported Bidirectional Standard JPS "
                    f"parameters: {sorted(unknown)}"
                )

            result = run_bidirectional_standard_jps(
                grid,
                start,
                goal,
                algorithm_name=label,
                heuristic_weight=float(
                    parameters.get(
                        "heuristic_weight",
                        1.0,
                    )
                ),
            )

        elif algorithm_name == "Smooth JPS":
            allowed = {
                "heuristic_weight",
                "smooth_tie_break",
            }

            unknown = set(parameters) - allowed

            if unknown:
                raise ValueError(
                    f"Unsupported Smooth JPS parameters: "
                    f"{sorted(unknown)}"
                )

            result = run_jps(
                grid,
                start,
                goal,
                algorithm_name=label,
                heuristic_weight=float(
                    parameters.get(
                        "heuristic_weight",
                        1.0,
                    )
                ),
                smooth_tie_break=bool(
                    parameters.get(
                        "smooth_tie_break",
                        True,
                    )
                ),
            )

        elif algorithm_name == "SA-JPS":
            allowed = {
                "turn_weight",
                "heuristic_weight",
            }

            unknown = set(parameters) - allowed

            if unknown:
                raise ValueError(
                    f"Unsupported SA-JPS parameters: "
                    f"{sorted(unknown)}"
                )

            result = run_sa_jps(
                grid,
                start,
                goal,
                algorithm_name=label,
                turn_weight=float(
                    parameters.get(
                        "turn_weight",
                        CONFIG["sa_turn_weight"],
                    )
                ),
                heuristic_weight=float(
                    parameters.get(
                        "heuristic_weight",
                        1.0,
                    )
                ),
            )


        elif algorithm_name == "Bidirectional SA-JPS":
            allowed = {
                "turn_weight",
                "heuristic_weight",
            }

            unknown = set(parameters) - allowed

            if unknown:
                raise ValueError(
                    "Unsupported Bidirectional SA-JPS "
                    f"parameters: {sorted(unknown)}"
                )

            result = run_bidirectional_sa_jps(
                grid,
                start,
                goal,
                algorithm_name=label,
                turn_weight=float(
                    parameters.get(
                        "turn_weight",
                        CONFIG["sa_turn_weight"],
                    )
                ),
                heuristic_weight=float(
                    parameters.get(
                        "heuristic_weight",
                        1.0,
                    )
                ),
            )

        elif algorithm_name == "Fast Weighted SA-JPS":
            allowed = {
                "turn_weight",
                "heuristic_weight",
            }

            unknown = set(parameters) - allowed

            if unknown:
                raise ValueError(
                    "Unsupported Fast Weighted SA-JPS "
                    f"parameters: {sorted(unknown)}"
                )

            result = run_sa_jps(
                grid,
                start,
                goal,
                algorithm_name=label,
                turn_weight=float(
                    parameters.get(
                        "turn_weight",
                        CONFIG["sa_turn_weight"],
                    )
                ),
                heuristic_weight=float(
                    parameters.get(
                        "heuristic_weight",
                        CONFIG[
                            "fast_sa_heuristic_weight"
                        ],
                    )
                ),
            )


        elif algorithm_name == "Bidirectional Fast Weighted SA-JPS":
            allowed = {
                "turn_weight",
                "heuristic_weight",
            }

            unknown = set(parameters) - allowed

            if unknown:
                raise ValueError(
                    "Unsupported Bidirectional Fast Weighted SA-JPS "
                    f"parameters: {sorted(unknown)}"
                )

            result = run_bidirectional_sa_jps(
                grid,
                start,
                goal,
                algorithm_name=label,
                turn_weight=float(
                    parameters.get(
                        "turn_weight",
                        CONFIG["sa_turn_weight"],
                    )
                ),
                heuristic_weight=float(
                    parameters.get(
                        "heuristic_weight",
                        CONFIG[
                            "fast_sa_heuristic_weight"
                        ],
                    )
                ),
            )

        elif algorithm_name == "JPS-Dijkstra Corridor":
            allowed = {
                "corridor_radius",
            }

            unknown = set(parameters) - allowed

            if unknown:
                raise ValueError(
                    "Unsupported JPS-Dijkstra Corridor "
                    f"parameters: {sorted(unknown)}"
                )

            result = run_jps_dijkstra_hybrid(
                grid,
                start,
                goal,
                corridor_radius=int(
                    parameters.get(
                        "corridor_radius",
                        CONFIG[
                            "hybrid_corridor_radius"
                        ],
                    )
                ),
            )

            # This algorithm sets its own fixed name internally.
            # Replace it with the requested configuration label.
            result.algorithm = label


        elif algorithm_name == "Fast Weighted SA-JPS-Dijkstra Corridor":
            allowed = {
                "corridor_radius",
                "turn_weight",
                "heuristic_weight",
            }

            unknown = set(parameters) - allowed

            if unknown:
                raise ValueError(
                    "Unsupported Fast Weighted SA-JPS-Dijkstra Corridor "
                    f"parameters: {sorted(unknown)}"
                )

            result = run_fast_sa_jps_dijkstra_hybrid(
                grid,
                start,
                goal,
                corridor_radius=int(
                    parameters.get(
                        "corridor_radius",
                        CONFIG["hybrid_corridor_radius"],
                    )
                ),
                turn_weight=float(
                    parameters.get(
                        "turn_weight",
                        CONFIG["sa_turn_weight"],
                    )
                ),
                heuristic_weight=float(
                    parameters.get(
                        "heuristic_weight",
                        CONFIG["fast_sa_heuristic_weight"],
                    )
                ),
                algorithm_name=label,
            )

        else:
            raise ValueError(
                f"Unsupported algorithm: {algorithm_name}"
            )

        # A* also sets its own fixed name internally.
        result.algorithm = label

        if algorithm_name in ('Standard JPS', 'Smooth JPS'):
            result.turn_weight = 0.0
            result.heuristic_weight = float(parameters.get('heuristic_weight', 1.0))

        # Attach configuration information to the result.
        result.base_algorithm = algorithm_name
        result.configuration_label = label
        result.parameters = parameters

        # Validate the generated path.
        if result.success:
            valid, message = validate_path(
                grid,
                result.path,
                start,
                goal,
            )

            if not valid:
                result.success = False
                result.note = (
                    f"Invalid path: {message}"
                )

        return result

    return [
        run_configuration(config)
        for config in run_configs
    ]


# In[16]:


# ==============================================================
# FAST WEIGHTED SA-JPS-DIJKSTRA CORRIDOR SMOKE TEST
# ==============================================================

def test_fast_sa_jps_dijkstra_corridor() -> pd.DataFrame:
    synthetic = np.ones((30, 40), dtype=bool)
    synthetic[5:25, 18] = False
    synthetic[14, 18] = True
    synthetic[4:20, 28] = False
    synthetic[9, 28] = True

    start = (2, 2)
    goal = (37, 27)

    results = run_all_algorithms(
        synthetic,
        start,
        goal,
        approaches=[
            "Standard JPS",
            "Fast Weighted SA-JPS",
            "JPS-Dijkstra Corridor",
            "Fast Weighted SA-JPS-Dijkstra Corridor",
        ],
    )

    table = pd.DataFrame(
        [result.metrics() for result in results]
    )

    return table[
        [
            "Algorithm",
            "Success",
            "Path length",
            "Turns",
            "Total turning angle deg",
            "Expanded nodes",
            "Generated successors",
            "Time ms",
            "Note",
        ]
    ]


# Run this before the complete benchmark:
# display(test_fast_sa_jps_dijkstra_corridor())


# ## 12. Final summary table
# 
# The summary reports medians for skewed runtime/count metrics and means for success and optimality.

# In[17]:


def build_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    """
    Build the dataset-level algorithm summary.

    For every numeric evaluation criterion, the table stores the mean,
    median, and sample standard deviation (ddof=1) whenever the source
    column is available.
    """
    successful = per_run[per_run["Success"].astype(bool)].copy()

    base = per_run.groupby(
        "Algorithm",
        observed=False,
    ).agg(
        Runs=("Success", "size"),
        Successful_runs=("Success", "sum"),
        Success_rate=("Success", "mean"),
    )

    metric_columns = [
        ("path_length", "Path length"),
        ("dataset_reference_gap_percent", "Optimality gap percent"),
        ("gap_vs_Astar_percent", "Gap vs A* percent"),
        ("time_ms", "Time ms"),
        ("expanded_nodes", "Expanded nodes"),
        ("generated_successors", "Generated successors"),
        ("queue_pushes", "Queue pushes"),
        ("jump_calls", "Jump calls"),
        ("turns", "Turns"),
        ("total_turning_angle_deg", "Total turning angle deg"),
        ("waypoints", "Waypoints"),
        ("search_objective", "Search objective"),
        ("turn_severity_45deg_units", "Turn severity 45deg units"),
    ]

    named_aggregations = {}

    for output_stem, source_column in metric_columns:
        if source_column not in successful.columns:
            continue

        named_aggregations[
            f"Mean_{output_stem}"
        ] = (source_column, "mean")

        named_aggregations[
            f"Median_{output_stem}"
        ] = (source_column, "median")

        named_aggregations[
            f"Std_{output_stem}"
        ] = (source_column, "std")

        named_aggregations[
            f"Min_{output_stem}"
        ] = (source_column, "min")

        named_aggregations[
            f"Max_{output_stem}"
        ] = (source_column, "max")

    numeric = (
        successful
        .groupby(
            "Algorithm",
            observed=False,
        )
        .agg(**named_aggregations)
    )

    summary = base.join(
        numeric,
        how="left",
    ).reset_index()

    summary["Success_rate"] *= 100.0

    # Retain the established column names used elsewhere in the notebook.
    aliases = {
        "Mean_dataset_reference_gap_percent":
            "Mean_dataset_reference_gap_percent",
        "Max_dataset_reference_gap_percent":
            "Max_dataset_reference_gap_percent",
        "Mean_gap_vs_Astar_percent":
            "Mean_gap_vs_Astar_percent",
        "Max_gap_vs_Astar_percent":
            "Max_gap_vs_Astar_percent",
    }

    for source_name, target_name in aliases.items():
        if (
            source_name in summary.columns
            and target_name not in summary.columns
        ):
            summary[target_name] = summary[source_name]

    return summary


# ## 13. Paired comparison against Standard JPS
# 
# Each scenario is paired with its own Standard JPS result, reducing distortion from differences in map difficulty.

# In[18]:


def paired_vs_standard(per_run: pd.DataFrame) -> pd.DataFrame:
    keys = ['Map name', 'Scenario index']
    successful = per_run[per_run['Success']].copy()
    baseline = successful[successful['Algorithm'].astype(str) == 'Standard JPS'][
        keys + ['Time ms', 'Expanded nodes', 'Path length', 'Turns']
    ].rename(columns={
        'Time ms': 'Baseline time ms',
        'Expanded nodes': 'Baseline expanded nodes',
        'Path length': 'Baseline path length',
        'Turns': 'Baseline turns',
    })
    paired = successful.merge(baseline, on=keys, how='inner')
    paired['Runtime change percent'] = 100.0 * (
        paired['Time ms'] - paired['Baseline time ms']
    ) / paired['Baseline time ms'].replace(0, np.nan)
    paired['Expanded nodes change percent'] = 100.0 * (
        paired['Expanded nodes'] - paired['Baseline expanded nodes']
    ) / paired['Baseline expanded nodes'].replace(0, np.nan)
    paired['Path length change percent'] = 100.0 * (
        paired['Path length'] - paired['Baseline path length']
    ) / paired['Baseline path length'].replace(0, np.nan)
    paired['Turns change percent'] = 100.0 * (
        paired['Turns'] - paired['Baseline turns']
    ) / paired['Baseline turns'].replace(0, np.nan)

    result = paired.groupby('Algorithm', observed=False).agg(
        Paired_scenarios=('Scenario index', 'size'),
        Median_runtime_change_percent=('Runtime change percent', 'median'),
        Median_expanded_nodes_change_percent=('Expanded nodes change percent', 'median'),
        Median_path_length_change_percent=('Path length change percent', 'median'),
        Median_turns_change_percent=('Turns change percent', 'median'),
    ).reset_index()
    return result


# ## 14. SA-JPS turn-weight sensitivity analysis
# 
# This experiment reruns SA-JPS for several `turn_weight` values. It quantifies the trade-off between geometric path length, turns, turning angle, expanded nodes, and runtime. A value of `0.0` reduces the composite objective to distance only.
# 

# In[19]:


def benchmark_sa_weight_sweep() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    weights = [float(value) for value in CONFIG['sa_turn_weight_sweep']]
    scenario_limit = CONFIG.get('scenarios_per_map')
    total = (
        len(selected_maps) * int(scenario_limit) * len(weights)
        if scenario_limit is not None else None
    )
    progress = tqdm(total=total, desc='SA-JPS weight sweep')

    for map_path in selected_maps:
        grid = read_map(map_path)
        scenario_path = locate_file(SCENARIOS_DIR, map_path.name + '.scen')
        scenarios = select_scenarios(
            read_scenario(scenario_path),
            CONFIG.get('scenarios_per_map'),
            str(CONFIG['scenario_selection']),
        ).reset_index(drop=True)

        for scenario_index, scenario in scenarios.iterrows():
            start = (int(scenario.start_x), int(scenario.start_y))
            goal = (int(scenario.goal_x), int(scenario.goal_y))
            reference_length = float(scenario.reference_length)

            for weight in weights:
                result = run_sa_jps(
                    grid, start, goal,
                    algorithm_name='SA-JPS',
                    turn_weight=weight,
                    heuristic_weight=1.0,
                )
                valid, message = validate_path(grid, result.path, start, goal) if result.success else (False, 'No path')
                if result.success and not valid:
                    result.success = False
                    result.note = f'Invalid path: {message}'

                metrics = result.metrics()
                measured_length = metrics['Path length']
                rows.append({
                    'Map name': map_path.name,
                    'Scenario index': int(scenario_index),
                    'Turn penalty weight': weight,
                    'Reference path length': reference_length,
                    'Gap vs dataset reference percent': (
                        100.0 * (float(measured_length) - reference_length) / reference_length
                        if result.success and reference_length > 0 else np.nan
                    ),
                    'Search objective': getattr(result, 'search_objective', np.nan),
                    'Turn severity 45deg units': getattr(result, 'turn_severity', np.nan),
                    **metrics,
                })
                progress.update(1)

    progress.close()
    return pd.DataFrame(rows)


from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import json
import os
import pandas as pd


def _benchmark_checkpoint_paths(
    cache_data_path: Path,
) -> Tuple[Path, Path]:
    """
    Return checkpoint data and metadata paths associated with the final cache.
    """
    token = active_dataset_name()
    base_stem = cache_data_path.stem
    if base_stem.lower().endswith(f"_{token.lower()}"):
        base_stem = base_stem[:-(len(token) + 1)]

    checkpoint_data_path = dataset_output_path(
        cache_data_path.with_name(
            f"{base_stem}_checkpoint.pkl"
        )
    )

    checkpoint_metadata_path = dataset_output_path(
        cache_data_path.with_name(
            f"{base_stem}_checkpoint_metadata.json"
        )
    )

    return checkpoint_data_path, checkpoint_metadata_path


def _atomic_pickle_save(
    frame: pd.DataFrame,
    destination: Path,
) -> None:
    """
    Save a DataFrame atomically.

    The temporary file is replaced only after writing succeeds, which reduces
    the chance of leaving a corrupted checkpoint after interruption.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    frame.to_pickle(temporary_path)

    os.replace(
        temporary_path,
        destination,
    )


def _atomic_json_save(
    payload: Dict[str, object],
    destination: Path,
) -> None:
    """Write JSON atomically."""
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    os.replace(
        temporary_path,
        destination,
    )


def _save_benchmark_checkpoint(
    rows: Sequence[Dict[str, object]],
    checkpoint_data_path: Path,
    checkpoint_metadata_path: Path,
    signature: Dict[str, object],
    *,
    completed_scenarios: int,
    completed_searches: int,
    current_map: Optional[str] = None,
) -> None:
    """Save partial benchmark rows and resume metadata."""
    frame = pd.DataFrame(rows)

    _atomic_pickle_save(
        frame,
        checkpoint_data_path,
    )

    metadata = {
        "signature": signature,
        "completed_scenarios": int(completed_scenarios),
        "completed_searches": int(completed_searches),
        "current_map": current_map,
        "rows": int(len(frame)),
    }

    _atomic_json_save(
        metadata,
        checkpoint_metadata_path,
    )


def _load_benchmark_checkpoint(
    checkpoint_data_path: Path,
    checkpoint_metadata_path: Path,
    signature: Dict[str, object],
) -> Optional[pd.DataFrame]:
    """
    Load a compatible partial benchmark.

    An incompatible checkpoint is ignored.
    """
    if (
        not checkpoint_data_path.exists()
        or not checkpoint_metadata_path.exists()
    ):
        return None

    try:
        with checkpoint_metadata_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            metadata = json.load(handle)

        if metadata.get("signature") != signature:
            print(
                "Existing benchmark checkpoint is incompatible "
                "with the current configuration and will be ignored."
            )
            return None

        frame = pd.read_pickle(
            checkpoint_data_path
        )

        if not isinstance(frame, pd.DataFrame):
            return None

        print()
        print("=" * 72)
        print("PARTIAL BENCHMARK CHECKPOINT LOADED")
        print("=" * 72)
        print(
            f"Checkpoint rows : {len(frame):,}"
        )
        print(
            f"Checkpoint file : "
            f"{checkpoint_data_path.resolve()}"
        )
        print("=" * 72)
        print()

        return frame

    except Exception as error:
        print(
            "Could not load benchmark checkpoint: "
            f"{error}"
        )
        return None


def _completed_execution_keys(
    checkpoint: pd.DataFrame,
) -> Set[Tuple[str, int, str]]:
    """
    Identify algorithm executions already stored in the checkpoint.
    """
    required = {
        "Map name",
        "Source scenario index",
        "Algorithm",
    }

    if (
        checkpoint is None
        or checkpoint.empty
        or not required.issubset(checkpoint.columns)
    ):
        return set()

    return {
        (
            str(row["Map name"]),
            int(row["Source scenario index"]),
            str(row["Algorithm"]),
        )
        for _, row in checkpoint.iterrows()
    }


def _delete_benchmark_checkpoint(
    checkpoint_data_path: Path,
    checkpoint_metadata_path: Path,
) -> None:
    """Remove checkpoint files after successful benchmark completion."""
    for path in (
        checkpoint_data_path,
        checkpoint_metadata_path,
    ):
        try:
            path.unlink(missing_ok=True)
        except TypeError:
            if path.exists():
                path.unlink()
                
                
                

def benchmark_dataset(
    *,
    approaches: Optional[Sequence[str]] = None,
    use_cache: Optional[bool] = None,
    force_recompute: Optional[bool] = None,
) -> Tuple[
    pd.DataFrame,
    Dict[
        Tuple[str, int],
        Tuple[
            np.ndarray,
            Point,
            Point,
            List[SearchResult],
        ],
    ],
]:
    """
    Benchmark the selected MovingAI dataset with persistent caching and
    detailed progress tracking.

    Parameters
    ----------
    approaches:
        Algorithms to benchmark. If None, all available algorithms are used.

        Example:
        approaches=[
            "Standard JPS",
            "SA-JPS",
            "Fast Weighted SA-JPS",
        ]

    use_cache:
        When True, load a compatible benchmark result from the persistent
        cache if it exists. The default value is read from
        CONFIG["use_benchmark_cache"].

    force_recompute:
        When True, ignore an existing compatible cache and recompute the
        benchmark. The default value is read from
        CONFIG["force_recompute_benchmark"].

    Returns
    -------
    frame:
        DataFrame containing one row for each algorithm execution.

    examples:
        Dictionary containing the first evaluated map/scenario and its paths.
        When results are loaded from cache, this dictionary is empty because
        path arrays are not stored in the benchmark cache.

    Progress information
    --------------------
    The function displays:

    - total maps;
    - total selected scenarios;
    - scenarios for every map;
    - selected algorithms;
    - total algorithm executions;
    - current map;
    - current scenario;
    - overall completed scenario count;
    - completed algorithm executions;
    - elapsed time;
    - estimated remaining time.
    """

    import time

    # ----------------------------------------------------------
    # Resolve settings
    # ----------------------------------------------------------
    selected_approaches = normalise_approaches(approaches)

    if use_cache is None:
        use_cache = bool(
            CONFIG.get("use_benchmark_cache", True)
        )

    if force_recompute is None:
        force_recompute = bool(
            CONFIG.get("force_recompute_benchmark", False)
        )

    # ----------------------------------------------------------
    # Construct cache identity
    # ----------------------------------------------------------
    signature = _benchmark_cache_signature(
        selected_approaches
    )

    cache_data_path, cache_metadata_path = (
        _benchmark_cache_paths(signature)
    )

    # ----------------------------------------------------------
    # Load compatible cached benchmark
    # ----------------------------------------------------------
    if use_cache and not force_recompute:
        cached = _load_cached_benchmark(
            cache_data_path,
            cache_metadata_path,
            signature,
        )

        if cached is not None:
            print()
            print("=" * 72)
            print("BENCHMARK LOADED FROM CACHE")
            print("=" * 72)
            print(
                f"Cache file          : "
                f"{cache_data_path.resolve()}"
            )
            print(
                f"Benchmark rows      : "
                f"{len(cached):,}"
            )

            if "Map name" in cached.columns:
                print(
                    f"Maps                : "
                    f"{cached['Map name'].nunique():,}"
                )

            if {
                "Map name",
                "Scenario index",
            }.issubset(cached.columns):
                cached_scenarios = (
                    cached[
                        [
                            "Map name",
                            "Scenario index",
                        ]
                    ]
                    .drop_duplicates()
                    .shape[0]
                )

                print(
                    f"Scenario iterations : "
                    f"{cached_scenarios:,}"
                )

            if "Algorithm" in cached.columns:
                cached_algorithms = (
                    cached["Algorithm"]
                    .astype(str)
                    .nunique()
                )

                print(
                    f"Algorithms          : "
                    f"{cached_algorithms:,}"
                )

            print("=" * 72)
            print()

            return cached, {}

    # ----------------------------------------------------------
    # Prepare all map/scenario selections before benchmarking
    #
    # This allows the exact total number of iterations to be
    # calculated before the progress bar starts.
    # ----------------------------------------------------------
    prepared_maps: List[
        Tuple[
            int,
            Path,
            np.ndarray,
            Path,
            pd.DataFrame,
            str,
            Dict[str, float],
        ]
    ] = []

    scenario_counts: List[int] = []

#     print()
    print("Preparing benchmark workload...")

    for map_index, map_path in enumerate(
        selected_maps,
        start=1,
    ):
        grid = read_map(map_path)
        map_type, map_features = get_map_type(grid)

        scenario_path = locate_file(
            SCENARIOS_DIR,
            map_path.name + ".scen",
        )

        all_scenarios = read_scenario(
            scenario_path
        )

        scenarios = select_scenarios(
            all_scenarios,
            CONFIG.get("scenarios_per_map"),
            str(CONFIG["scenario_selection"]),
        ).reset_index(drop=True)

        prepared_maps.append(
            (
                map_index,
                map_path,
                grid,
                scenario_path,
                scenarios,
                map_type,
                map_features,
            )
        )

        scenario_counts.append(len(scenarios))

    # ----------------------------------------------------------
    # Calculate exact workload
    # ----------------------------------------------------------
    num_maps = len(prepared_maps)

    total_scenario_iterations = int(
        sum(scenario_counts)
    )

    num_algorithms = len(selected_approaches)

    total_search_executions = (
        total_scenario_iterations
        * num_algorithms
    )

    # ----------------------------------------------------------
    # Display workload summary
    # ----------------------------------------------------------
    print()
    print("=" * 72)
    print("BENCHMARK WORKLOAD")
    print("=" * 72)
    print(
        f"Dataset             : "
        f"{CONFIG.get('dataset_name', 'Unknown')}"
    )
    print(
        f"Maps                : "
        f"{num_maps:,}"
    )
    print(
        f"Scenario iterations : "
        f"{total_scenario_iterations:,}"
    )
    print(
        f"Algorithms          : "
        f"{num_algorithms:,}"
    )
    print(
        f"Search executions   : "
        f"{total_search_executions:,}"
    )
    print("-" * 72)

    for (
        map_index,
        map_path,
        _grid,
        _scenario_path,
        scenarios,
        map_type,
        _map_features,
    ) in prepared_maps:
        print(
            f"[{map_index:>3}/{num_maps}] "
            f"{map_path.name:<40} "
            f"{len(scenarios):>8,} scenarios "
            f"[{map_type}]"
        )

    print("-" * 72)
    print("Selected approaches:")

    for algorithm_index, algorithm_name in enumerate(
        selected_approaches,
        start=1,
    ):
        print(
            f"  [{algorithm_index}/{num_algorithms}] "
            f"{algorithm_name}"
        )

    print("=" * 72)
    print()

    if total_scenario_iterations == 0:
        raise RuntimeError(
            "No scenarios were selected for benchmarking. "
            "Check CONFIG['scenarios_per_map'], "
            "CONFIG['scenario_selection'], and the dataset files."
        )

    if num_algorithms == 0:
        raise RuntimeError(
            "No benchmark approaches were selected."
        )

    # ----------------------------------------------------------
    # Benchmark containers
    # ----------------------------------------------------------
    rows: List[Dict[str, object]] = []

    examples: Dict[
        Tuple[str, int],
        Tuple[
            np.ndarray,
            Point,
            Point,
            List[SearchResult],
        ],
    ] = {}

    journal = SearchJournal(
        cache_data_path.with_suffix('.sqlite3'), signature,
        reset=bool(force_recompute),
    )
    rows = journal.rows()
    if not rows and not force_recompute:
        legacy = _load_benchmark_checkpoint(
            *_benchmark_checkpoint_paths(cache_data_path), signature
        )
        if legacy is not None:
            rows = legacy.to_dict('records')
            for row in rows:
                journal.save(row)
    completed = _completed_execution_keys(pd.DataFrame(rows))
    print(f'Resuming with {len(completed):,} completed searches.')

    # ----------------------------------------------------------
    # Progress state
    # ----------------------------------------------------------
    scenario_counter = 0
    search_counter = len(completed)

    benchmark_start_time = time.perf_counter()

    progress = tqdm(
        total=total_scenario_iterations,
        desc="Benchmark",
        unit="scenario",
        dynamic_ncols=True,
        leave=True,
    )

    try:
        # ------------------------------------------------------
        # Process every map
        # ------------------------------------------------------
        for (
            map_index,
            map_path,
            grid,
            scenario_path,
            scenarios,
            map_type,
            map_features,
        ) in prepared_maps:

            number_of_map_scenarios = len(scenarios)

            # --------------------------------------------------
            # Process every selected scenario
            # --------------------------------------------------
            for local_scenario_number, (
                scenario_index,
                scenario,
            ) in enumerate(
                scenarios.iterrows(),
                start=1,
            ):
                scenario_counter += 1

                source_scenario_index = int(
                    getattr(
                        scenario,
                        "source_scenario_index",
                        scenario_index,
                    )
                )

                pending = [
                    item for item in selected_approaches
                    if (map_path.name, source_scenario_index,
                        str(item.get('label', item['algorithm'])) if isinstance(item, dict) else item)
                    not in completed
                ]
                if not pending:
                    progress.update(1)
                    continue

                start = (
                    int(scenario.start_x),
                    int(scenario.start_y),
                )

                goal = (
                    int(scenario.goal_x),
                    int(scenario.goal_y),
                )

                reference_length = float(
                    scenario.reference_length
                )

                # ----------------------------------------------
                # Calculate elapsed time and ETA before search
                # ----------------------------------------------
                elapsed_seconds = (
                    time.perf_counter()
                    - benchmark_start_time
                )

                completed_before_current = (
                    scenario_counter - 1
                )

                if completed_before_current > 0:
                    average_seconds_per_scenario = (
                        elapsed_seconds
                        / completed_before_current
                    )

                    remaining_scenarios = (
                        total_scenario_iterations
                        - completed_before_current
                    )

                    estimated_remaining_seconds = (
                        average_seconds_per_scenario
                        * remaining_scenarios
                    )
                else:
                    estimated_remaining_seconds = 0.0

                elapsed_text = time.strftime(
                    "%H:%M:%S",
                    time.gmtime(elapsed_seconds),
                )

                eta_text = (
                    time.strftime(
                        "%H:%M:%S",
                        time.gmtime(
                            estimated_remaining_seconds
                        ),
                    )
                    if completed_before_current > 0
                    else "--:--:--"
                )

                progress.set_description(
                    f"Map {map_index}/{num_maps}"
                )

                progress.set_postfix(
                    {
                        "name": map_path.stem[:18],
                        "scenario": (
                            f"{local_scenario_number}/"
                            f"{number_of_map_scenarios}"
                        ),
                        "overall": (
                            f"{scenario_counter}/"
                            f"{total_scenario_iterations}"
                        ),
                        "searches": (
                            f"{search_counter}/"
                            f"{total_search_executions}"
                        ),
                        "elapsed": elapsed_text,
                        "ETA": eta_text,
                    },
                    refresh=True,
                )

                # ----------------------------------------------
                # Run selected algorithms
                # ----------------------------------------------
                # Execute lazily so each method is committed before the next starts.
                algorithm_results = []
                pending_results = (
                    run_all_algorithms(grid, start, goal, approaches=[item])[0]
                    for item in pending
                )

                # ----------------------------------------------
                # Save first evaluated example
                # ----------------------------------------------
                if not examples:
                    examples[
                        (
                            map_path.name,
                            source_scenario_index,
                        )
                    ] = (
                        grid,
                        start,
                        goal,
                        algorithm_results,
                    )

                # ----------------------------------------------
                # Store metrics for each algorithm
                # ----------------------------------------------
                for result in pending_results:
                    algorithm_results.append(result)
                    search_counter += 1
                    metrics = result.metrics()

                    measured_length = metrics[
                        "Path length"
                    ]

                    optimality_gap = (
                        100.0
                        * (
                            float(measured_length)
                            - reference_length
                        )
                        / reference_length
                        if (
                            result.success
                            and reference_length > 0
                        )
                        else np.nan
                    )

                    rows.append(
                        {
                            "Map name": map_path.name,
                            "Map type": map_type,
                            "Map width": grid.shape[1],
                            "Map height": grid.shape[0],
                            "Obstacle percentage":
                                obstacle_percentage(grid),
                            "Obstacle density":
                                map_features["Obstacle density"],
                            "Free-cell ratio":
                                map_features["Free-cell ratio"],
                            "Dead-end ratio":
                                map_features["Dead-end ratio"],
                            "Corridor ratio":
                                map_features["Corridor ratio"],
                            "Junction ratio":
                                map_features["Junction ratio"],
                            "Obstacle-edge density":
                                map_features["Obstacle-edge density"],
                            "Mean axis visibility":
                                map_features["Mean axis visibility"],
                            "Connected components":
                                map_features["Connected components"],
                            "Largest component ratio":
                                map_features["Largest component ratio"],

                            "Scenario index":
                                int(scenario_index),

                            "Source scenario index":
                                source_scenario_index,

                            "Start x": start[0],
                            "Start y": start[1],
                            "Goal x": goal[0],
                            "Goal y": goal[1],

                            "Reference path length":
                                reference_length,

                            "Optimality gap percent":
                                optimality_gap,

                            "Search objective":
                                getattr(
                                    result,
                                    "search_objective",
                                    np.nan,
                                ),

                            "Turn severity 45deg units":
                                getattr(
                                    result,
                                    "turn_severity",
                                    np.nan,
                                ),

                            "Turn penalty weight":
                                getattr(
                                    result,
                                    "turn_weight",
                                    np.nan,
                                ),

                            "Heuristic weight":
                                getattr(
                                    result,
                                    "heuristic_weight",
                                    np.nan,
                                ),

                            **metrics,
                        }
                    )

                    journal.save(rows[-1])
                    completed.add((map_path.name, source_scenario_index, str(rows[-1]['Algorithm'])))

                # ----------------------------------------------
                # Complete one scenario iteration
                # ----------------------------------------------
                progress.update(1)

                # ----------------------------------------------
                # Update progress after completed search
                # ----------------------------------------------
                elapsed_seconds = (
                    time.perf_counter()
                    - benchmark_start_time
                )

                average_seconds_per_scenario = (
                    elapsed_seconds
                    / scenario_counter
                )

                remaining_scenarios = (
                    total_scenario_iterations
                    - scenario_counter
                )

                estimated_remaining_seconds = (
                    average_seconds_per_scenario
                    * remaining_scenarios
                )

                elapsed_text = time.strftime(
                    "%H:%M:%S",
                    time.gmtime(elapsed_seconds),
                )

                eta_text = time.strftime(
                    "%H:%M:%S",
                    time.gmtime(
                        estimated_remaining_seconds
                    ),
                )

                progress.set_postfix(
                    {
                        "name": map_path.stem[:18],
                        "scenario": (
                            f"{local_scenario_number}/"
                            f"{number_of_map_scenarios}"
                        ),
                        "overall": (
                            f"{scenario_counter}/"
                            f"{total_scenario_iterations}"
                        ),
                        "searches": (
                            f"{search_counter}/"
                            f"{total_search_executions}"
                        ),
                        "elapsed": elapsed_text,
                        "ETA": eta_text,
                    },
                    refresh=True,
                )

    finally:
        journal.close()
        progress.close()

    # ----------------------------------------------------------
    # Create benchmark DataFrame
    # ----------------------------------------------------------
    frame = pd.DataFrame(rows)

    if not frame.empty:
        # ------------------------------------------------------
        # Compute reference path lengths from A*
        # ------------------------------------------------------
        astar_reference = (
            frame[
                frame["Algorithm"] == "A*"
            ][
                [
                    "Map name",
                    "Scenario index",
                    "Path length",
                ]
            ]
            .rename(
                columns={
                    "Path length":
                        "A* reference length"
                }
            )
        )

        frame = frame.merge(
            astar_reference,
            on=[
                "Map name",
                "Scenario index",
            ],
            how="left",
        )

        # ------------------------------------------------------
        # Compute gap relative to A*
        #
        # When A* is not selected, this column remains NaN.
        # ------------------------------------------------------
        frame["Gap vs A* percent"] = (
            100.0
            * (
                frame["Path length"]
                - frame["A* reference length"]
            )
            / frame[
                "A* reference length"
            ].replace(0, np.nan)
        )

        # ------------------------------------------------------
        # Set consistent algorithm ordering
        # ------------------------------------------------------
        frame["Algorithm"] = pd.Categorical(
            frame["Algorithm"],
            categories=ALGORITHM_ORDER,
            ordered=True,
        )

        frame = frame.sort_values(
            [
                "Map name",
                "Scenario index",
                "Algorithm",
            ]
        ).reset_index(drop=True)

    # ----------------------------------------------------------
    # Save persistent benchmark cache
    # ----------------------------------------------------------
    if use_cache:
        _save_benchmark_cache(
            frame,
            cache_data_path,
            cache_metadata_path,
            signature,
        )

    # ----------------------------------------------------------
    # Final benchmark report
    # ----------------------------------------------------------
    total_elapsed_seconds = (
        time.perf_counter()
        - benchmark_start_time
    )

    total_elapsed_text = time.strftime(
        "%H:%M:%S",
        time.gmtime(total_elapsed_seconds),
    )

    average_scenario_time = (
        total_elapsed_seconds
        / total_scenario_iterations
        if total_scenario_iterations > 0
        else 0.0
    )

    average_search_time = (
        total_elapsed_seconds
        / search_counter
        if search_counter > 0
        else 0.0
    )

    print()
    print("=" * 72)
    print("BENCHMARK COMPLETED")
    print("=" * 72)
    print(
        f"Maps processed       : "
        f"{num_maps:,}/{num_maps:,}"
    )
    print(
        f"Scenario iterations  : "
        f"{scenario_counter:,}/"
        f"{total_scenario_iterations:,}"
    )
    print(
        f"Search executions    : "
        f"{search_counter:,}/"
        f"{total_search_executions:,}"
    )
    print(
        f"Generated rows       : "
        f"{len(frame):,}"
    )
    print(
        f"Total elapsed time   : "
        f"{total_elapsed_text}"
    )
    print(
        f"Average per scenario : "
        f"{average_scenario_time:.6f} s"
    )
    print(
        f"Average per search   : "
        f"{average_search_time:.6f} s"
    )

    if use_cache:
        print(
            f"Cache data           : "
            f"{cache_data_path.resolve()}"
        )
        print(
            f"Cache metadata       : "
            f"{cache_metadata_path.resolve()}"
        )

    print("=" * 72)
    print()

    return frame, examples


# ## 15. Plots

# In[20]:


def corresponding_std_column(
    metric_name: str,
) -> Optional[str]:
    """
    Return the standard-deviation column associated with a summary metric.

    Examples
    --------
    Mean_time_ms   -> Std_time_ms
    Median_time_ms -> Std_time_ms
    Mean turns     -> Std turns
    Median turns   -> Std turns
    """
    if metric_name.startswith("Mean_"):
        return "Std_" + metric_name[len("Mean_"):]

    if metric_name.startswith("Median_"):
        return "Std_" + metric_name[len("Median_"):]

    if metric_name.startswith("Mean "):
        return "Std " + metric_name[len("Mean "):]

    if metric_name.startswith("Median "):
        return "Std " + metric_name[len("Median "):]

    return None

def save_bar_plot(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    filename: str,
    ascending: bool = True,
    output_dir: Optional[Path] = None,
) -> None:
    """Save a sorted bar plot and the exact plotted rows as CSV."""
    if x not in frame.columns:
        raise KeyError(f"Column {x!r} was not found.")
    if y not in frame.columns:
        raise KeyError(f"Column {y!r} was not found.")

    plot_columns = [x, y]

    std_column = corresponding_std_column(y)

    if (
        std_column is not None
        and std_column in frame.columns
    ):
        plot_columns.append(std_column)

    plot_frame = frame[plot_columns].copy()
    plot_frame[y] = pd.to_numeric(plot_frame[y], errors="coerce")
    plot_frame = (
        plot_frame.dropna(subset=[y])
        .sort_values(by=y, ascending=ascending, kind="stable")
        .reset_index(drop=True)
    )
    if plot_frame.empty:
        print(f"No valid data available for {y!r}.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(plot_frame[x].astype(str), plot_frame[y])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y.replace("_", " "))
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, plot_frame[y]):
        ax.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    target_dir = Path(output_dir) if output_dir is not None else RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_output_path(target_dir / filename)
    fig.savefig(path, dpi=180, bbox_inches="tight")

    if CONFIG.get("save_plot_data_csv", True):
        data_path = save_plot_data_csv(plot_frame, path)
        print(f"Saved plot data: {data_path.resolve()}")

    plt.show()
    plt.close(fig)
    print(f"Saved figure: {path.resolve()}")


# In[ ]:





# In[21]:


from pathlib import Path
import re


ALGORITHM_FILE_NAMES = {
    "A*": "Astar",
    "Standard JPS": "Standard_JPS",
    "Bidirectional Standard JPS": "Bidirectional_Standard_JPS",
    "Smooth JPS": "Smooth_JPS",
    "SA-JPS": "SA_JPS",
    "Bidirectional SA-JPS": "Bidirectional_SA_JPS",
    "Fast SA-JPS": "Fast_SA_JPS",
    "Fast Weighted SA-JPS": "Fast_Weighted_SA_JPS",
    "Bidirectional Fast Weighted SA-JPS": "Bidirectional_Fast_Weighted_SA_JPS",
    "JPS-Dijkstra": "JPS_Dijkstra",
    "JPS-Dijkstra Corridor": "JPS_Dijkstra_Corridor",
    "Fast Weighted SA-JPS-Dijkstra Corridor": "Fast_Weighted_SA_JPS_Dijkstra_Corridor",
}


def safe_filename(name: str) -> str:
    """
    Convert an arbitrary algorithm name into a valid Windows filename.
    """

    mapped_name = ALGORITHM_FILE_NAMES.get(name)

    if mapped_name is not None:
        return mapped_name

    # Replace characters that are invalid in Windows filenames.
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", str(name))

    # Replace spaces and hyphens.
    safe_name = safe_name.replace(" ", "_")
    safe_name = safe_name.replace("-", "_")

    # Remove repeated underscores.
    safe_name = re.sub(r"_+", "_", safe_name)

    # Remove leading/trailing periods, spaces, and underscores.
    safe_name = safe_name.strip(" ._")

    if not safe_name:
        safe_name = "algorithm"

    return safe_name


def plot_paths(
    grid: np.ndarray,
    start: Point,
    goal: Point,
    results: Sequence[SearchResult],
    filename: str = "example_paths.png",
) -> Dict[str, object]:
    """
    Save one combined comparison figure and one separate image
    for each successful algorithm.

    Output structure:

    results/
        example_paths.png
        example_paths/
            Astar.png
            Standard_JPS.png
            Smooth_JPS.png
            SA_JPS.png
            Fast_SA_JPS.png
            JPS_Dijkstra_Corridor.png
    """

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    successful = [
        result
        for result in results
        if result.success and result.path
    ]

    if not successful:
        print("No successful paths are available to plot.")
        return {"combined": None, "individual": {}}

    filename_path = Path(filename)

    # Ensure the combined figure has an image extension.
    if filename_path.suffix == "":
        filename_path = filename_path.with_suffix(".png")

    combined_output = dataset_output_path(results_dir / filename_path)
    combined_output.parent.mkdir(parents=True, exist_ok=True)

    # Each set of individual images is stored beside the combined image.
    image_folder = combined_output.parent / filename_path.stem
    image_folder.mkdir(parents=True, exist_ok=True)

    # Determine common crop limits so every algorithm is shown
    # using exactly the same map region.
    all_x = [start[0], goal[0]]
    all_y = [start[1], goal[1]]

    for result in successful:
        all_x.extend(point[0] for point in result.path)
        all_y.extend(point[1] for point in result.path)

    padding = 10

    x_min = max(0, min(all_x) - padding)
    x_max = min(grid.shape[1] - 1, max(all_x) + padding)

    y_min = max(0, min(all_y) - padding)
    y_max = min(grid.shape[0] - 1, max(all_y) + padding)

    # ------------------------------------------------------------------
    # Combined figure
    # ------------------------------------------------------------------

    fig_width = max(6, 6 * len(successful))

    fig, axes = plt.subplots(
        1,
        len(successful),
        figsize=(fig_width, 6),
        squeeze=False,
    )

    for ax, result in zip(axes[0], successful):
        xs = [point[0] for point in result.path]
        ys = [point[1] for point in result.path]

        path_len = path_length(result.path)
        turns, total_turn_angle, maximum_turn_angle = path_turn_metrics(
            result.path
        )

        ax.imshow(
            ~grid,
            cmap="gray_r",
            origin="upper",
            interpolation="nearest",
        )

        ax.plot(
            xs,
            ys,
            linewidth=2,
            label=result.algorithm,
        )

        ax.scatter(
            [start[0]],
            [start[1]],
            marker="o",
            s=70,
            label="Start",
            zorder=3,
        )

        ax.scatter(
            [goal[0]],
            [goal[1]],
            marker="X",
            s=90,
            label="Goal",
            zorder=3,
        )

        ax.set_title(
            f"{result.algorithm}\n"
            f"L={path_len:.3f}, "
            f"turns={turns}, "
            f"angle={total_turn_angle:.1f}°"
        )

        ax.set_xlim(x_min, x_max)

        # Reverse the limits because image row zero is at the top.
        ax.set_ylim(y_max, y_min)

        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(False)

        # ------------------------------------------------------------------
        # Individual figure
        # ------------------------------------------------------------------

        fig_single, ax_single = plt.subplots(
            figsize=(8, 8)
        )

        ax_single.imshow(
            ~grid,
            cmap="gray_r",
            origin="upper",
            interpolation="nearest",
        )

        ax_single.plot(
            xs,
            ys,
            linewidth=2,
            label=result.algorithm,
        )

        ax_single.scatter(
            [start[0]],
            [start[1]],
            marker="o",
            s=90,
            label="Start",
            zorder=3,
        )

        ax_single.scatter(
            [goal[0]],
            [goal[1]],
            marker="X",
            s=110,
            label="Goal",
            zorder=3,
        )

        ax_single.set_title(
            f"{result.algorithm}\n"
            f"Path length = {path_len:.3f}\n"
            f"Turns = {turns}, "
            f"total angle = {total_turn_angle:.1f}°, "
            f"maximum angle = {maximum_turn_angle:.1f}°"
        )

        ax_single.set_xlim(x_min, x_max)
        ax_single.set_ylim(y_max, y_min)
        ax_single.set_aspect("equal")
        ax_single.set_xlabel("X")
        ax_single.set_ylabel("Y")
        ax_single.grid(False)
        ax_single.legend(loc="best")

        algorithm_name = safe_filename(
            result.algorithm
        )

        single_file = dataset_output_path(
            image_folder / f"{algorithm_name}.png"
        )

        if CONFIG.get("save_plot_data_csv", True):
            individual_plot_data = pd.DataFrame({
                "Algorithm": result.algorithm,
                "Point index": np.arange(len(result.path), dtype=int),
                "X": xs,
                "Y": ys,
                "Is start": [
                    index == 0 for index in range(len(result.path))
                ],
                "Is goal": [
                    index == len(result.path) - 1
                    for index in range(len(result.path))
                ],
            })
            individual_data_path = save_plot_data_csv(
                individual_plot_data,
                single_file,
                label="path_data",
            )
            print(
                f"Saved individual path data: "
                f"{individual_data_path.resolve()}"
            )

        fig_single.tight_layout()

        fig_single.savefig(
            single_file,
            dpi=600,
            bbox_inches="tight",
        )

        plt.close(fig_single)

        print(
            f"Saved individual image: "
            f"{single_file.resolve()}"
        )

    # Hide any unused axes, although normally the count matches.
    for ax in axes[0][len(successful):]:
        ax.axis("off")

    fig.tight_layout()

    if CONFIG.get("save_plot_data_csv", True):
        combined_rows = []
        for result in successful:
            for point_index, (x_value, y_value) in enumerate(result.path):
                combined_rows.append({
                    "Algorithm": result.algorithm,
                    "Point index": point_index,
                    "X": x_value,
                    "Y": y_value,
                    "Is start": point_index == 0,
                    "Is goal": point_index == len(result.path) - 1,
                })
        combined_data_path = save_plot_data_csv(
            pd.DataFrame(combined_rows),
            combined_output,
            label="path_data",
        )
        print(f"Saved combined path data: {combined_data_path.resolve()}")

    fig.savefig(
        combined_output,
        dpi=600,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

#     print(
#         f"\nCombined figure saved to:\n"
#         f"{combined_output.resolve()}"
#     )

#     print(
#         f"\nIndividual figures saved under:\n"
#         f"{image_folder.resolve()}"
#     )


    return {
        "combined": combined_output,
        "individual": {
            result.algorithm: dataset_output_path(image_folder / f"{safe_filename(result.algorithm)}.png")
            for result in successful
        },
    }


# ## 16. Interpretation guide
# 
# Use the exported tables to answer four separate questions:
# 
# 1. **Search efficiency:** compare runtime, expanded nodes, generated successors, queue pushes, and jump calls.
# 2. **Geometric optimality:** compare path length and gap versus A* and the MovingAI reference.
# 3. **Smoothness:** compare turns, cumulative turning angle, maximum turn angle, and waypoints.
# 4. **SA-JPS trade-off:** inspect the turn-weight sweep. Increasing `turn_weight` should reduce turning complexity, but may increase geometric length or search effort.
# 
# ### Distinguishing the methods
# 
# - **Smooth JPS** uses turns only to resolve equal-distance alternatives. It should preserve distance optimality but may show little change when exact ties are rare.
# - **SA-JPS** places turn severity directly in the accumulated objective. It can deliberately accept a slightly longer route when the reduction in turning cost compensates for that distance.
# - **Fast Weighted SA-JPS** additionally weights the distance heuristic. It is intended to reduce expansions and runtime, but it may increase both path-length and composite-objective gaps.
# - **JPS–Dijkstra Corridor** is a local refinement baseline and is expected to be smoother but considerably slower.
# 
# ### Important validity limitation
# 
# Because standard JPS pruning was derived for a distance-only uniform grid, SA-JPS should be described as an experimental direction-aware extension. For a formal optimality claim under the composite objective, compare it against a full direction-aware A* or Dijkstra search using the same turn penalty. The present notebook supports empirical path, speed, and smoothness comparisons but does not prove composite-cost optimality.
# 

# In[22]:


def package_results(
    results_dir: Path = RESULTS_DIR,
    archive_name: str = "astar_jps_benchmark_results",
) -> Path:
    """Create a ZIP archive containing every generated result file."""
    import shutil

    archive_base = dataset_output_path(
        Path(archive_name)
    )
    archive = shutil.make_archive(
        str(archive_base),
        "zip",
        root_dir=Path(results_dir),
    )
    archive_path = Path(archive).resolve()
    print(f"Created {archive_path}")
    return archive_path


# ## 17. One-click experiment controller
# 
# All functions are defined above. Edit only the following `CONFIG` cell, then run the final cell.
# 

# ## 7. Ablation study
# 
# The following functions run one-factor-at-a-time parameter sweeps and a component comparison. Enable them in the final `CONFIG` cell.
# 

# In[23]:



# ==============================================================
# ABLATION STUDY
# ==============================================================

def _ablation_selected_scenarios(map_path: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load one selected map and the same scenario subset used by the main benchmark."""
    grid = read_map(map_path)
    scenario_path = locate_file(SCENARIOS_DIR, map_path.name + '.scen')
    scenarios = select_scenarios(
        read_scenario(scenario_path),
        CONFIG.get('scenarios_per_map'),
        str(CONFIG['scenario_selection']),
    ).reset_index(drop=True)
    return grid, scenarios


def _validate_ablation_result(
    result: SearchResult,
    grid: np.ndarray,
    start: Point,
    goal: Point,
) -> SearchResult:
    """Mark an ablation result unsuccessful when its returned path is invalid."""
    if result.success:
        valid, message = validate_path(grid, result.path, start, goal)
        if not valid:
            result.success = False
            result.note = f'Invalid path: {message}'
    return result


def _ablation_row(
    *,
    study: str,
    parameter_name: str,
    parameter_value: float,
    map_name: str,
    scenario_index: int,
    reference_length: float,
    astar_length: float,
    result: SearchResult,
) -> Dict[str, object]:
    """Convert one ablation execution into a consistent result row."""
    metrics = result.metrics()
    measured_length = metrics['Path length']

    dataset_gap = (
        100.0 * (float(measured_length) - reference_length) / reference_length
        if result.success and reference_length > 0 else np.nan
    )
    astar_gap = (
        100.0 * (float(measured_length) - astar_length) / astar_length
        if result.success and astar_length > 0 else np.nan
    )

    return {
        'Study': study,
        'Parameter': parameter_name,
        'Parameter value': parameter_value,
        'Map name': map_name,
        'Scenario index': int(scenario_index),
        'Reference path length': reference_length,
        'A* reference length': astar_length,
        'Gap vs dataset reference percent': dataset_gap,
        'Gap vs A* percent': astar_gap,
        'Search objective': getattr(result, 'search_objective', np.nan),
        'Turn severity 45deg units': getattr(result, 'turn_severity', np.nan),
        **metrics,
    }


def run_parameter_ablation(
    *,
    study: str,
    parameter_name: str,
    values: Sequence[float],
    runner,
) -> pd.DataFrame:
    """
    Run a one-factor-at-a-time parameter sweep.

    `runner` must have the signature:
        runner(grid, start, goal, parameter_value) -> SearchResult
    """
    clean_values = list(dict.fromkeys(values))
    if not clean_values:
        print(f"Skipping {study}: no parameter values were supplied.")
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    scenario_limit = CONFIG.get('scenarios_per_map')
    total = (
        len(selected_maps) * int(scenario_limit) * len(clean_values)
        if scenario_limit is not None else None
    )
    progress = tqdm(total=total, desc=study)

    for map_path in selected_maps:
        grid, scenarios = _ablation_selected_scenarios(map_path)

        for scenario_index, scenario in scenarios.iterrows():
            start = (int(scenario.start_x), int(scenario.start_y))
            goal = (int(scenario.goal_x), int(scenario.goal_y))
            reference_length = float(scenario.reference_length)

            # Compute A* once per scenario and reuse it for every parameter value.
            astar_result = _validate_ablation_result(
                run_astar(grid, start, goal),
                grid,
                start,
                goal,
            )
            astar_length = (
                path_length(astar_result.path)
                if astar_result.success else np.nan
            )

            for value in clean_values:
                result = runner(grid, start, goal, value)
                result = _validate_ablation_result(
                    result, grid, start, goal
                )

                rows.append(_ablation_row(
                    study=study,
                    parameter_name=parameter_name,
                    parameter_value=float(value),
                    map_name=map_path.name,
                    scenario_index=int(scenario_index),
                    reference_length=reference_length,
                    astar_length=astar_length,
                    result=result,
                ))
                progress.update(1)

    progress.close()
    return pd.DataFrame(rows)


def summarize_ablation(
    per_run: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate one ablation sweep by parameter value.

    Mean, median, and sample standard deviation are stored for every
    available numeric criterion.
    """
    if per_run.empty:
        return pd.DataFrame()

    grouping = [
        "Study",
        "Parameter",
        "Parameter value",
    ]

    base = per_run.groupby(
        grouping,
        as_index=False,
        observed=False,
    ).agg(
        Runs=("Success", "size"),
        Successful_runs=("Success", "sum"),
        Success_rate=("Success", "mean"),
    )

    successful = per_run[
        per_run["Success"].astype(bool)
    ].copy()

    metric_columns = [
        ("time_ms", "Time ms"),
        ("expanded_nodes", "Expanded nodes"),
        ("generated_successors", "Generated successors"),
        ("queue_pushes", "Queue pushes"),
        ("jump_calls", "Jump calls"),
        ("path_length", "Path length"),
        ("gap_vs_Astar_percent", "Gap vs A* percent"),
        (
            "dataset_reference_gap_percent",
            "Gap vs dataset reference percent",
        ),
        ("turns", "Turns"),
        (
            "total_turning_angle_deg",
            "Total turning angle deg",
        ),
        ("waypoints", "Waypoints"),
        ("search_objective", "Search objective"),
        (
            "turn_severity_45deg_units",
            "Turn severity 45deg units",
        ),
    ]

    named_aggregations = {}

    for output_stem, source_column in metric_columns:
        if source_column not in successful.columns:
            continue

        named_aggregations[
            f"Mean_{output_stem}"
        ] = (source_column, "mean")

        named_aggregations[
            f"Median_{output_stem}"
        ] = (source_column, "median")

        named_aggregations[
            f"Std_{output_stem}"
        ] = (source_column, "std")

        named_aggregations[
            f"Min_{output_stem}"
        ] = (source_column, "min")

        named_aggregations[
            f"Max_{output_stem}"
        ] = (source_column, "max")

    numeric = (
        successful
        .groupby(
            grouping,
            as_index=False,
            observed=False,
        )
        .agg(**named_aggregations)
    )

    summary = base.merge(
        numeric,
        on=grouping,
        how="left",
    )

    summary["Success_rate"] *= 100.0

    return (
        summary
        .sort_values("Parameter value")
        .reset_index(drop=True)
    )

def save_ablation_line_plot(
    summary: pd.DataFrame,
    y: str,
    title: str,
    filename: str,
    output_dir: Path,
) -> None:
    """Save a parameter-sensitivity line plot."""
    if summary.empty or y not in summary.columns:
        print(f"Skipping ablation plot; missing data: {y}")
        return

    plot_columns = [
        "Parameter value",
        y,
    ]

    std_column = corresponding_std_column(
        y
    )

    if (
        std_column is not None
        and std_column in summary.columns
    ):
        plot_columns.append(std_column)

    plot_frame = summary[
        plot_columns
    ].copy()
    plot_frame[y] = pd.to_numeric(plot_frame[y], errors='coerce')
    plot_frame = (
        plot_frame.dropna(subset=[y])
        .sort_values('Parameter value')
        .reset_index(drop=True)
    )
    if plot_frame.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        plot_frame['Parameter value'],
        plot_frame[y],
        marker='o',
        linewidth=1.8,
    )
    for x_value, y_value in zip(
        plot_frame['Parameter value'],
        plot_frame[y],
    ):
        ax.annotate(
            f'{y_value:.3g}',
            (x_value, y_value),
            xytext=(0, 6),
            textcoords='offset points',
            ha='center',
            fontsize=8,
        )

    parameter_label = str(summary['Parameter'].iloc[0])
    ax.set_title(title)
    ax.set_xlabel(parameter_label)
    ax.set_ylabel(y.replace('_', ' '))
    ax.grid(alpha=0.25)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_output_path(output_dir / filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    if CONFIG.get('save_plot_data_csv', True):
        data_path = save_plot_data_csv(plot_frame, path)
        print(f'Saved plot data: {data_path.resolve()}')
    plt.show()
    plt.close(fig)
    print(f'Saved {path.resolve()}')


def save_ablation_plots(
    summary: pd.DataFrame,
    output_dir: Path,
    prefix: str,
) -> None:
    """Save the standard plots for one ablation sweep."""
    plot_specs = [
        ('Median_time_ms', 'Median runtime', f'{prefix}_runtime.png'),
        (
            'Median_expanded_nodes',
            'Median expanded nodes',
            f'{prefix}_expanded_nodes.png',
        ),
        (
            'Mean_gap_vs_Astar_percent',
            'Mean path-length gap versus A*',
            f'{prefix}_gap_vs_astar.png',
        ),
        ('Median_turns', 'Median number of turns', f'{prefix}_turns.png'),
        (
            'Median_total_turning_angle_deg',
            'Median total turning angle',
            f'{prefix}_turning_angle.png',
        ),
        ('Median_waypoints', 'Median waypoints', f'{prefix}_waypoints.png'),
    ]
    for metric, title, filename in plot_specs:
        save_ablation_line_plot(
            summary,
            metric,
            title,
            filename,
            output_dir,
        )


def run_component_ablation() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare the principal algorithmic components using the configured values.

    Configurations:
      - Standard JPS
      - Weighted JPS
      - SA-JPS
      - Weighted SA-JPS
      - JPS-Dijkstra Corridor
    """
    configurations = [
        (
            'Standard JPS',
            lambda grid, start, goal: run_jps(
                grid,
                start,
                goal,
                algorithm_name='Standard JPS',
                heuristic_weight=1.0,
                smooth_tie_break=False,
            ),
        ),
        (
            'Weighted JPS',
            lambda grid, start, goal: run_jps(
                grid,
                start,
                goal,
                algorithm_name='Weighted JPS',
                heuristic_weight=float(CONFIG['fast_jps_weight']),
                smooth_tie_break=False,
            ),
        ),
        (
            'SA-JPS',
            lambda grid, start, goal: run_sa_jps(
                grid,
                start,
                goal,
                algorithm_name='SA-JPS',
                turn_weight=float(CONFIG['sa_turn_weight']),
                heuristic_weight=1.0,
            ),
        ),
        (
            'Weighted SA-JPS',
            lambda grid, start, goal: run_sa_jps(
                grid,
                start,
                goal,
                algorithm_name='Weighted SA-JPS',
                turn_weight=float(CONFIG['sa_turn_weight']),
                heuristic_weight=float(CONFIG['fast_sa_heuristic_weight']),
            ),
        ),
        (
            'JPS-Dijkstra Corridor',
            lambda grid, start, goal: run_jps_dijkstra_hybrid(
                grid,
                start,
                goal,
                corridor_radius=int(CONFIG['hybrid_corridor_radius']),
            ),
        ),
    ]

    rows: List[Dict[str, object]] = []
    scenario_limit = CONFIG.get('scenarios_per_map')
    total = (
        len(selected_maps) * int(scenario_limit) * len(configurations)
        if scenario_limit is not None else None
    )
    progress = tqdm(total=total, desc='Component ablation')

    for map_path in selected_maps:
        grid, scenarios = _ablation_selected_scenarios(map_path)

        for scenario_index, scenario in scenarios.iterrows():
            start = (int(scenario.start_x), int(scenario.start_y))
            goal = (int(scenario.goal_x), int(scenario.goal_y))
            reference_length = float(scenario.reference_length)

            astar_result = _validate_ablation_result(
                run_astar(grid, start, goal),
                grid, start, goal,
            )
            astar_length = (
                path_length(astar_result.path)
                if astar_result.success else np.nan
            )

            for configuration_name, runner in configurations:
                result = _validate_ablation_result(
                    runner(grid, start, goal),
                    grid, start, goal,
                )
                metrics = result.metrics()
                measured_length = metrics['Path length']

                rows.append({
                    'Configuration': configuration_name,
                    'Map name': map_path.name,
                    'Scenario index': int(scenario_index),
                    'Reference path length': reference_length,
                    'A* reference length': astar_length,
                    'Gap vs A* percent': (
                        100.0 * (float(measured_length) - astar_length)
                        / astar_length
                        if (
                            result.success
                            and np.isfinite(astar_length)
                            and astar_length > 0
                        )
                        else np.nan
                    ),
                    'Search objective': getattr(
                        result, 'search_objective', np.nan
                    ),
                    'Turn severity 45deg units': getattr(
                        result, 'turn_severity', np.nan
                    ),
                    **metrics,
                })
                progress.update(1)

    progress.close()
    per_run = pd.DataFrame(rows)

    base = per_run.groupby(
        'Configuration',
        as_index=False,
        observed=False,
    ).agg(
        Runs=('Success', 'size'),
        Successful_runs=('Success', 'sum'),
        Success_rate=('Success', 'mean'),
    )

    successful = per_run[per_run['Success']].copy()
    numeric = successful.groupby(
        'Configuration',
        as_index=False,
        observed=False,
    ).agg(
        Median_time_ms=('Time ms', 'median'),
        Median_expanded_nodes=('Expanded nodes', 'median'),
        Median_generated_successors=('Generated successors', 'median'),
        Median_path_length=('Path length', 'median'),
        Mean_gap_vs_Astar_percent=('Gap vs A* percent', 'mean'),
        Median_turns=('Turns', 'median'),
        Median_total_turning_angle_deg=(
            'Total turning angle deg', 'median'
        ),
        Median_waypoints=('Waypoints', 'median'),
    )

    summary = base.merge(numeric, on='Configuration', how='left')
    summary['Success_rate'] *= 100.0

    order = [name for name, _ in configurations]
    summary['Configuration'] = pd.Categorical(
        summary['Configuration'],
        order,
        ordered=True,
    )
    summary = summary.sort_values('Configuration').reset_index(drop=True)
    summary['Configuration'] = summary['Configuration'].astype(str)

    return per_run, summary


def save_component_ablation_plots(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save sorted bar plots for the component-removal comparison."""
    specs = [
        ('Median_time_ms', 'Component ablation: median runtime',
         'component_runtime.png'),
        ('Median_expanded_nodes',
         'Component ablation: median expanded nodes',
         'component_expanded_nodes.png'),
        ('Mean_gap_vs_Astar_percent',
         'Component ablation: mean path-length gap versus A*',
         'component_gap_vs_astar.png'),
        ('Median_turns', 'Component ablation: median turns',
         'component_turns.png'),
        ('Median_total_turning_angle_deg',
         'Component ablation: median turning angle',
         'component_turning_angle.png'),
    ]
    for metric, title, filename in specs:
        if metric not in summary.columns:
            continue
        save_bar_plot(
            summary,
            'Configuration',
            metric,
            title,
            filename,
            ascending=True,
            output_dir=output_dir,
        )


def run_ablation_study() -> Dict[str, object]:
    """Run all enabled ablation experiments and save their outputs."""
    ablation_dir = RESULTS_DIR / 'ablation'
    ablation_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, object] = {}

    studies = [
        {
            'enabled': CONFIG.get('ablation_fast_jps_weight', True),
            'key': 'fast_jps_weight',
            'study': 'Fast JPS heuristic weight',
            'parameter': 'Heuristic weight',
            'values': CONFIG.get('ablation_fast_jps_weights', []),
            'runner': lambda grid, start, goal, value: run_jps(
                grid,
                start,
                goal,
                algorithm_name='Weighted JPS',
                heuristic_weight=float(value),
                smooth_tie_break=False,
            ),
        },
        {
            'enabled': CONFIG.get('ablation_sa_turn_weight', True),
            'key': 'sa_turn_weight',
            'study': 'SA-JPS turn penalty',
            'parameter': 'Turn penalty weight',
            'values': CONFIG.get(
                'ablation_sa_turn_weights',
                CONFIG.get('sa_turn_weight_sweep', []),
            ),
            'runner': lambda grid, start, goal, value: run_sa_jps(
                grid,
                start,
                goal,
                algorithm_name='SA-JPS',
                turn_weight=float(value),
                heuristic_weight=1.0,
            ),
        },
        {
            'enabled': CONFIG.get(
                'ablation_fast_sa_heuristic_weight', True
            ),
            'key': 'fast_sa_heuristic_weight',
            'study': 'Fast SA-JPS heuristic weight',
            'parameter': 'Heuristic weight',
            'values': CONFIG.get(
                'ablation_fast_sa_heuristic_weights', []
            ),
            'runner': lambda grid, start, goal, value: run_sa_jps(
                grid,
                start,
                goal,
                algorithm_name='Weighted SA-JPS',
                turn_weight=float(CONFIG['sa_turn_weight']),
                heuristic_weight=float(value),
            ),
        },
        {
            'enabled': CONFIG.get('ablation_corridor_radius', True),
            'key': 'corridor_radius',
            'study': 'Hybrid corridor radius',
            'parameter': 'Corridor radius',
            'values': CONFIG.get('ablation_corridor_radii', []),
            'runner': lambda grid, start, goal, value:
                run_jps_dijkstra_hybrid(
                    grid,
                    start,
                    goal,
                    corridor_radius=int(value),
                ),
        },
    ]

    for specification in studies:
        if not specification['enabled']:
            print(f"Skipping {specification['study']} (disabled).")
            continue

        key = str(specification['key'])
        study_dir = ablation_dir / key
        study_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nRunning ablation: {specification['study']}")
        per_run = run_parameter_ablation(
            study=str(specification['study']),
            parameter_name=str(specification['parameter']),
            values=specification['values'],
            runner=specification['runner'],
        )
        summary = summarize_ablation(per_run)

        per_run_path = dataset_output_path(study_dir / f'{key}_per_run.csv')
        summary_path = dataset_output_path(study_dir / f'{key}_summary.csv')
        per_run.to_csv(per_run_path, index=False)
        summary.to_csv(summary_path, index=False)

        save_ablation_plots(summary, study_dir, key)

        outputs[f'{key}_per_run'] = per_run
        outputs[f'{key}_summary'] = summary

#         print(f'Saved {per_run_path.resolve()}')
#         print(f'Saved {summary_path.resolve()}')
        if CONFIG.get('display_tables', True):
            display(summary)

    if CONFIG.get('ablation_component_comparison', True):
        print('\nRunning component ablation comparison...')
        component_per_run, component_summary = run_component_ablation()
        component_dir = ablation_dir / 'components'
        component_dir.mkdir(parents=True, exist_ok=True)

        component_per_run_path = (
            dataset_output_path(component_dir / 'component_ablation_per_run.csv')
        )
        component_summary_path = (
            dataset_output_path(component_dir / 'component_ablation_summary.csv')
        )
        component_per_run.to_csv(component_per_run_path, index=False)
        component_summary.to_csv(component_summary_path, index=False)
        save_component_ablation_plots(component_summary, component_dir)

        outputs['component_per_run'] = component_per_run
        outputs['component_summary'] = component_summary

#         print(f'Saved {component_per_run_path.resolve()}')
#         print(f'Saved {component_summary_path.resolve()}')
        if CONFIG.get('display_tables', True):
            display(component_summary)

    return outputs


# ## Per-map benchmark analysis
# 
# The final run uses every matching 256×256 map when `map_limit=None`. It saves a per-map summary table and one line graph for each metric. Every graph uses map names on the x-axis and one line for each algorithm.
# 

# In[24]:


# ==============================================================
# NON-PARAMETRIC STATISTICAL COMPARISON
# Shapiro-Wilk, paired Wilcoxon, Friedman, and Nemenyi
# ==============================================================

from itertools import combinations

from scipy.stats import (
    friedmanchisquare,
    rankdata,
    shapiro,
    studentized_range,
    wilcoxon,
)


DEFAULT_STATISTICAL_METRICS = [
    "Time ms",
    "Expanded nodes",
    "Generated successors",
    "Path length",
    "Gap vs A* percent",
    "Turns",
    "Total turning angle deg",
    "Waypoints",
]


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down family-wise error correction."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))

    if len(finite_indices) == 0:
        return adjusted

    ordered_indices = finite_indices[
        np.argsort(values[finite_indices])
    ]
    number = len(ordered_indices)
    running_max = 0.0

    for order, index in enumerate(ordered_indices):
        candidate = min(
            1.0,
            (number - order) * values[index],
        )
        running_max = max(running_max, candidate)
        adjusted[index] = running_max

    return adjusted


def _paired_metric_matrix(
    frame: pd.DataFrame,
    metric: str,
    *,
    algorithms: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Create complete scenario blocks: one row per map/scenario and one column
    per algorithm. Only successful finite observations are retained.
    """
    required = {
        "Map name",
        "Scenario index",
        "Algorithm",
        "Success",
        metric,
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(
            f"Cannot analyse {metric!r}; missing columns: {sorted(missing)}"
        )

    usable = frame[
        frame["Success"].astype(bool)
    ].copy()
    usable[metric] = pd.to_numeric(
        usable[metric],
        errors="coerce",
    )
    usable = usable[
        np.isfinite(usable[metric])
    ]

    if algorithms is not None:
        selected = [str(value) for value in algorithms]
        usable = usable[
            usable["Algorithm"].astype(str).isin(selected)
        ]
    else:
        selected = [
            str(value)
            for value in ALGORITHM_ORDER
            if str(value) in set(
                usable["Algorithm"].astype(str)
            )
        ]

    matrix = usable.pivot_table(
        index=["Map name", "Scenario index"],
        columns="Algorithm",
        values=metric,
        aggfunc="first",
        observed=False
    )

    existing = [
        algorithm
        for algorithm in selected
        if algorithm in matrix.columns
    ]
    matrix = matrix.reindex(columns=existing)
    return matrix.dropna(axis=0, how="any")


def run_shapiro_tests(
    per_run: pd.DataFrame,
    *,
    metrics: Optional[Sequence[str]] = None,
    algorithms: Optional[Sequence[str]] = None,
    group_label: str = "All maps",
) -> pd.DataFrame:
    """
    Test normality of paired differences for every algorithm pair.

    Shapiro-Wilk is applied to paired differences rather than to unrelated raw
    samples because the subsequent comparison is paired by map/scenario.
    """
    selected_metrics = list(
        metrics or DEFAULT_STATISTICAL_METRICS
    )
    rows: List[Dict[str, object]] = []

    for metric in selected_metrics:
        if metric not in per_run.columns:
            continue

        matrix = _paired_metric_matrix(
            per_run,
            metric,
            algorithms=algorithms,
        )

        for algorithm_a, algorithm_b in combinations(
            matrix.columns,
            2,
        ):
            differences = (
                matrix[algorithm_a]
                - matrix[algorithm_b]
            ).to_numpy(dtype=float)
            differences = differences[
                np.isfinite(differences)
            ]

            if len(differences) < 3:
                statistic = np.nan
                p_value = np.nan
                note = "Fewer than 3 paired observations"
            elif np.allclose(differences, differences[0]):
                statistic = np.nan
                p_value = np.nan
                note = "Paired differences are constant"
            else:
                # scipy warns for very large N; use a deterministic maximum of
                # 5000 observations, which is also the range for which the
                # p-value is best calibrated.
                tested = differences
                if len(tested) > 5000:
                    indices = np.linspace(
                        0,
                        len(tested) - 1,
                        5000,
                        dtype=int,
                    )
                    tested = tested[indices]
                    note = "Deterministic 5000-observation subset"
                else:
                    note = ""

                statistic, p_value = shapiro(tested)

            rows.append(
                {
                    "Group": group_label,
                    "Metric": metric,
                    "Algorithm A": str(algorithm_a),
                    "Algorithm B": str(algorithm_b),
                    "Paired observations": int(len(differences)),
                    "Shapiro W": float(statistic)
                    if np.isfinite(statistic)
                    else np.nan,
                    "p-value": float(p_value)
                    if np.isfinite(p_value)
                    else np.nan,
                    "Normal at alpha=0.05": bool(p_value >= 0.05)
                    if np.isfinite(p_value)
                    else np.nan,
                    "Note": note,
                }
            )

    return pd.DataFrame(rows)


def run_pairwise_wilcoxon(
    per_run: pd.DataFrame,
    *,
    metrics: Optional[Sequence[str]] = None,
    algorithms: Optional[Sequence[str]] = None,
    group_label: str = "All maps",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Perform all paired Wilcoxon signed-rank comparisons.

    Holm-adjusted p-values control family-wise error within each metric/group.
    Rank-biserial correlation is reported as an effect size. A positive effect
    means Algorithm A tends to have larger metric values than Algorithm B.
    """
    selected_metrics = list(
        metrics or DEFAULT_STATISTICAL_METRICS
    )
    rows: List[Dict[str, object]] = []

    for metric in selected_metrics:
        if metric not in per_run.columns:
            continue

        matrix = _paired_metric_matrix(
            per_run,
            metric,
            algorithms=algorithms,
        )
        metric_rows: List[Dict[str, object]] = []

        for algorithm_a, algorithm_b in combinations(
            matrix.columns,
            2,
        ):
            values_a = matrix[algorithm_a].to_numpy(dtype=float)
            values_b = matrix[algorithm_b].to_numpy(dtype=float)
            differences = values_a - values_b
            nonzero = differences[~np.isclose(differences, 0.0)]

            if len(values_a) == 0:
                statistic = np.nan
                p_value = np.nan
                rank_biserial = np.nan
                note = "No complete paired observations"
            elif len(nonzero) == 0:
                statistic = 0.0
                p_value = 1.0
                rank_biserial = 0.0
                note = "All paired differences are zero"
            else:
                statistic, p_value = wilcoxon(
                    values_a,
                    values_b,
                    alternative="two-sided",
                    zero_method="wilcox",
                    correction=False,
                    method="auto",
                )

                absolute_ranks = rankdata(
                    np.abs(nonzero),
                    method="average",
                )
                positive_sum = float(
                    absolute_ranks[nonzero > 0].sum()
                )
                negative_sum = float(
                    absolute_ranks[nonzero < 0].sum()
                )
                denominator = positive_sum + negative_sum
                rank_biserial = (
                    (positive_sum - negative_sum) / denominator
                    if denominator > 0
                    else 0.0
                )
                note = ""

            median_a = float(np.median(values_a)) if len(values_a) else np.nan
            median_b = float(np.median(values_b)) if len(values_b) else np.nan
            median_difference = (
                float(np.median(differences))
                if len(differences)
                else np.nan
            )

            if np.isfinite(median_difference):
                if median_difference < 0:
                    lower_metric_algorithm = str(algorithm_a)
                elif median_difference > 0:
                    lower_metric_algorithm = str(algorithm_b)
                else:
                    lower_metric_algorithm = "Tie"
            else:
                lower_metric_algorithm = ""

            metric_rows.append(
                {
                    "Group": group_label,
                    "Metric": metric,
                    "Algorithm A": str(algorithm_a),
                    "Algorithm B": str(algorithm_b),
                    "Paired observations": int(len(values_a)),
                    "Median A": median_a,
                    "Median B": median_b,
                    "Median paired difference A-B": median_difference,
                    "Wilcoxon statistic": float(statistic)
                    if np.isfinite(statistic)
                    else np.nan,
                    "Raw p-value": float(p_value)
                    if np.isfinite(p_value)
                    else np.nan,
                    "Rank-biserial correlation": float(rank_biserial)
                    if np.isfinite(rank_biserial)
                    else np.nan,
                    "Lower median metric": lower_metric_algorithm,
                    "Note": note,
                }
            )

        if metric_rows:
            adjusted = _holm_adjust(
                [
                    row["Raw p-value"]
                    for row in metric_rows
                ]
            )
            for row, adjusted_p in zip(metric_rows, adjusted):
                row["Holm-adjusted p-value"] = (
                    float(adjusted_p)
                    if np.isfinite(adjusted_p)
                    else np.nan
                )
                row[f"Significant at alpha={alpha:g}"] = (
                    bool(adjusted_p < alpha)
                    if np.isfinite(adjusted_p)
                    else np.nan
                )
            rows.extend(metric_rows)

    return pd.DataFrame(rows)


def _nemenyi_from_complete_matrix(
    matrix: pd.DataFrame,
    *,
    higher_is_better: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate Nemenyi pairwise comparisons from complete-block average ranks.

    Returns:
        average-rank table
        pairwise Nemenyi table
    """
    values = matrix.to_numpy(dtype=float)
    number_of_blocks, number_of_algorithms = values.shape

    if number_of_blocks < 2 or number_of_algorithms < 3:
        return pd.DataFrame(), pd.DataFrame()

    ranked_rows = np.vstack(
        [
            rankdata(
                -row if higher_is_better else row,
                method="average",
            )
            for row in values
        ]
    )
    average_ranks = ranked_rows.mean(axis=0)

    rank_table = pd.DataFrame(
        {
            "Algorithm": list(matrix.columns.astype(str)),
            "Average rank": average_ranks,
        }
    ).sort_values(
        "Average rank"
    ).reset_index(drop=True)

    standard_error = math.sqrt(
        number_of_algorithms
        * (number_of_algorithms + 1)
        / (6.0 * number_of_blocks)
    )

    rows: List[Dict[str, object]] = []
    for first_index, second_index in combinations(
        range(number_of_algorithms),
        2,
    ):
        rank_difference = abs(
            average_ranks[first_index]
            - average_ranks[second_index]
        )
        q_statistic = (
            rank_difference / standard_error
            if standard_error > 0
            else np.nan
        )
        p_value = (
            float(
                studentized_range.sf(
                    q_statistic * math.sqrt(2.0),
                    number_of_algorithms,
                    np.inf,
                )
            )
            if np.isfinite(q_statistic)
            else np.nan
        )

        rows.append(
            {
                "Algorithm A": str(matrix.columns[first_index]),
                "Algorithm B": str(matrix.columns[second_index]),
                "Average rank A": float(average_ranks[first_index]),
                "Average rank B": float(average_ranks[second_index]),
                "Absolute rank difference": float(rank_difference),
                "Nemenyi q": float(q_statistic)
                if np.isfinite(q_statistic)
                else np.nan,
                "Nemenyi p-value": p_value,
            }
        )

    return rank_table, pd.DataFrame(rows)


def run_friedman_nemenyi(
    per_run: pd.DataFrame,
    *,
    metrics: Optional[Sequence[str]] = None,
    algorithms: Optional[Sequence[str]] = None,
    group_label: str = "All maps",
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run a Friedman omnibus test and Nemenyi post-hoc comparisons.

    Only complete map/scenario blocks with one valid observation from every
    selected algorithm are used. Nemenyi is run only when Friedman is
    significant.
    """
    selected_metrics = list(
        metrics or DEFAULT_STATISTICAL_METRICS
    )
    omnibus_rows: List[Dict[str, object]] = []
    rank_frames: List[pd.DataFrame] = []
    nemenyi_frames: List[pd.DataFrame] = []

    for metric in selected_metrics:
        if metric not in per_run.columns:
            continue

        matrix = _paired_metric_matrix(
            per_run,
            metric,
            algorithms=algorithms,
        )
        algorithm_count = matrix.shape[1]
        block_count = matrix.shape[0]

        if algorithm_count < 3 or block_count < 2:
            omnibus_rows.append(
                {
                    "Group": group_label,
                    "Metric": metric,
                    "Complete blocks": int(block_count),
                    "Algorithms": int(algorithm_count),
                    "Friedman chi-square": np.nan,
                    "Degrees of freedom": max(algorithm_count - 1, 0),
                    "p-value": np.nan,
                    "Kendall W": np.nan,
                    f"Significant at alpha={alpha:g}": np.nan,
                    "Note": "Requires at least 3 algorithms and 2 complete blocks",
                }
            )
            continue

        samples = [
            matrix[column].to_numpy(dtype=float)
            for column in matrix.columns
        ]

        if np.allclose(
            matrix.to_numpy(dtype=float),
            matrix.iloc[:, [0]].to_numpy(dtype=float),
        ):
            statistic = 0.0
            p_value = 1.0
        else:
            statistic, p_value = friedmanchisquare(*samples)

        kendall_w = float(
            statistic
            / (
                block_count
                * (algorithm_count - 1)
            )
        ) if algorithm_count > 1 and block_count > 0 else np.nan

        significant = bool(p_value < alpha)
        omnibus_rows.append(
            {
                "Group": group_label,
                "Metric": metric,
                "Complete blocks": int(block_count),
                "Algorithms": int(algorithm_count),
                "Friedman chi-square": float(statistic),
                "Degrees of freedom": int(algorithm_count - 1),
                "p-value": float(p_value),
                "Kendall W": kendall_w,
                f"Significant at alpha={alpha:g}": significant,
                "Note": "",
            }
        )

        rank_table, nemenyi_table = (
            _nemenyi_from_complete_matrix(matrix)
        )

        if not rank_table.empty:
            rank_table.insert(0, "Metric", metric)
            rank_table.insert(0, "Group", group_label)
            rank_table["Complete blocks"] = int(block_count)
            rank_frames.append(rank_table)

        if significant and not nemenyi_table.empty:
            nemenyi_table.insert(0, "Metric", metric)
            nemenyi_table.insert(0, "Group", group_label)
            nemenyi_table["Complete blocks"] = int(block_count)
            nemenyi_table[
                f"Significant at alpha={alpha:g}"
            ] = (
                nemenyi_table["Nemenyi p-value"] < alpha
            )
            nemenyi_frames.append(nemenyi_table)

    omnibus = pd.DataFrame(omnibus_rows)
    ranks = (
        pd.concat(rank_frames, ignore_index=True)
        if rank_frames
        else pd.DataFrame()
    )
    nemenyi = (
        pd.concat(nemenyi_frames, ignore_index=True)
        if nemenyi_frames
        else pd.DataFrame()
    )
    return omnibus, ranks, nemenyi


def run_statistical_analysis(
    per_run: pd.DataFrame,
    *,
    metrics: Optional[Sequence[str]] = None,
    algorithms: Optional[Sequence[str]] = None,
    include_map_types: bool = True,
    alpha: float = 0.05,
    output_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run all statistical tests overall and separately for each map type.

    The analysis unit is a paired map/scenario block. This prevents scenarios
    from different maps or algorithms from being mismatched.
    """
    selected_metrics = list(
        metrics or DEFAULT_STATISTICAL_METRICS
    )
    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else RESULTS_DIR / "statistics"
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    groups: List[Tuple[str, pd.DataFrame]] = [
        ("All maps", per_run)
    ]

    if include_map_types and "Map type" in per_run.columns:
        for map_type, group in per_run.groupby(
            "Map type",
            observed=True,
        ):
            groups.append(
                (f"Map type: {map_type}", group.copy())
            )

    shapiro_frames: List[pd.DataFrame] = []
    wilcoxon_frames: List[pd.DataFrame] = []
    friedman_frames: List[pd.DataFrame] = []
    rank_frames: List[pd.DataFrame] = []
    nemenyi_frames: List[pd.DataFrame] = []

    for group_label, group_frame in groups:
        shapiro_frames.append(
            run_shapiro_tests(
                group_frame,
                metrics=selected_metrics,
                algorithms=algorithms,
                group_label=group_label,
            )
        )
        wilcoxon_frames.append(
            run_pairwise_wilcoxon(
                group_frame,
                metrics=selected_metrics,
                algorithms=algorithms,
                group_label=group_label,
                alpha=alpha,
            )
        )
        friedman, ranks, nemenyi = run_friedman_nemenyi(
            group_frame,
            metrics=selected_metrics,
            algorithms=algorithms,
            group_label=group_label,
            alpha=alpha,
        )
        friedman_frames.append(friedman)
        if not ranks.empty:
            rank_frames.append(ranks)
        if not nemenyi.empty:
            nemenyi_frames.append(nemenyi)

    outputs = {
        "shapiro": pd.concat(
            shapiro_frames,
            ignore_index=True,
        ) if shapiro_frames else pd.DataFrame(),
        "wilcoxon": pd.concat(
            wilcoxon_frames,
            ignore_index=True,
        ) if wilcoxon_frames else pd.DataFrame(),
        "friedman": pd.concat(
            friedman_frames,
            ignore_index=True,
        ) if friedman_frames else pd.DataFrame(),
        "average_ranks": pd.concat(
            rank_frames,
            ignore_index=True,
        ) if rank_frames else pd.DataFrame(),
        "nemenyi": pd.concat(
            nemenyi_frames,
            ignore_index=True,
        ) if nemenyi_frames else pd.DataFrame(),
    }

    filenames = {
        "shapiro": "shapiro_wilk_paired_differences.csv",
        "wilcoxon": "wilcoxon_pairwise_holm.csv",
        "friedman": "friedman_omnibus.csv",
        "average_ranks": "friedman_average_ranks.csv",
        "nemenyi": "nemenyi_posthoc.csv",
    }
    for name, frame in outputs.items():
        path = dataset_output_path(target_dir / filenames[name])
        frame.to_csv(path, index=False)
#         print(f"Saved {path.resolve()}")

    return outputs


def summarize_by_map_type(
    per_run: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate algorithm performance separately for each map type.

    Mean, median, and sample standard deviation are stored for every
    available numeric evaluation criterion.
    """
    if "Map type" not in per_run.columns:
        raise KeyError(
            "The benchmark frame has no 'Map type' column."
        )

    successful = per_run[
        per_run["Success"].astype(bool)
    ].copy()

    grouping = [
        "Map type",
        "Algorithm",
    ]

    aggregation = {
        "Runs": ("Success", "size"),
        "Success rate": ("Success", "mean"),
    }

    metric_columns = [
        ("time ms", "Time ms"),
        ("expanded nodes", "Expanded nodes"),
        (
            "generated successors",
            "Generated successors",
        ),
        ("queue pushes", "Queue pushes"),
        ("jump calls", "Jump calls"),
        ("path length", "Path length"),
        (
            "gap vs A* percent",
            "Gap vs A* percent",
        ),
        ("turns", "Turns"),
        (
            "turning angle deg",
            "Total turning angle deg",
        ),
        ("waypoints", "Waypoints"),
        (
            "search objective",
            "Search objective",
        ),
        (
            "turn severity 45deg units",
            "Turn severity 45deg units",
        ),
    ]

    for label, source_column in metric_columns:
        if source_column not in successful.columns:
            continue

        aggregation[
            f"Mean {label}"
        ] = (source_column, "mean")

        aggregation[
            f"Median {label}"
        ] = (source_column, "median")

        aggregation[
            f"Std {label}"
        ] = (source_column, "std")

        aggregation[
            f"Min {label}"
        ] = (source_column, "min")

        aggregation[
            f"Max {label}"
        ] = (source_column, "max")

    summary = (
        successful
        .groupby(
            grouping,
            observed=True,
        )
        .agg(**aggregation)
        .reset_index()
    )

    summary["Success rate"] *= 100.0

    return summary

def create_map_type_rankings(
    map_type_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Add within-map-type ranks for the principal lower-is-better metrics."""
    ranked = map_type_summary.copy()

    metric_rank_pairs = [
        ("Median time ms", "Runtime rank"),
        ("Median expanded nodes", "Expansion rank"),
        ("Median path length", "Path-length rank"),
        ("Mean gap vs A* percent", "Gap rank"),
        ("Median turns", "Turn rank"),
        ("Median turning angle deg", "Turning-angle rank"),
    ]

    for metric, rank_column in metric_rank_pairs:
        if metric in ranked.columns:
            ranked[rank_column] = (
                ranked.groupby(
                    "Map type",
                    observed=True,
                )[metric]
                .rank(
                    method="average",
                    ascending=True,
                )
            )

    return ranked


# In[25]:


# ==============================================================
# EFFECT-SIZE TABLES AND CRITICAL-DIFFERENCE (CD) DIAGRAMS
# ==============================================================

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import studentized_range


def classify_rank_biserial_effect(value: float) -> str:
    """Interpret the absolute rank-biserial correlation."""
    if not np.isfinite(value):
        return ""
    magnitude = abs(float(value))
    if magnitude < 0.10:
        return "Negligible"
    if magnitude < 0.30:
        return "Small"
    if magnitude < 0.50:
        return "Medium"
    return "Large"


def classify_kendall_w(value: float) -> str:
    """Interpret Kendall's W for the Friedman omnibus test."""
    if not np.isfinite(value):
        return ""
    magnitude = abs(float(value))
    if magnitude < 0.10:
        return "Negligible"
    if magnitude < 0.30:
        return "Weak"
    if magnitude < 0.50:
        return "Moderate"
    if magnitude < 0.70:
        return "Strong"
    return "Very strong"


def build_wilcoxon_effect_size_table(
    wilcoxon_results: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Build a publication-ready pairwise effect-size table.

    Direction:
      positive rank-biserial correlation -> Algorithm A tends to have
      larger metric values than Algorithm B;
      negative -> Algorithm A tends to have smaller values.

    For the default benchmark metrics, smaller values are preferable.
    """
    if wilcoxon_results is None or wilcoxon_results.empty:
        return pd.DataFrame()

    frame = wilcoxon_results.copy()
    effect_column = "Rank-biserial correlation"
    adjusted_column = "Holm-adjusted p-value"

    required = {
        "Group",
        "Metric",
        "Algorithm A",
        "Algorithm B",
        effect_column,
        adjusted_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(
            "Cannot build effect-size table; missing columns: "
            f"{sorted(missing)}"
        )

    frame["Absolute effect size"] = (
        pd.to_numeric(frame[effect_column], errors="coerce").abs()
    )
    frame["Effect magnitude"] = frame[effect_column].apply(
        classify_rank_biserial_effect
    )
    frame["Statistically significant"] = (
        pd.to_numeric(frame[adjusted_column], errors="coerce") < alpha
    )

    def preferred_algorithm(row: pd.Series) -> str:
        difference = pd.to_numeric(
            pd.Series([row.get("Median paired difference A-B")]),
            errors="coerce",
        ).iloc[0]
        if not np.isfinite(difference):
            return ""
        if difference < 0:
            return str(row["Algorithm A"])
        if difference > 0:
            return str(row["Algorithm B"])
        return "Tie"

    frame["Preferred algorithm (lower metric)"] = frame.apply(
        preferred_algorithm,
        axis=1,
    )

    columns = [
        "Group",
        "Metric",
        "Algorithm A",
        "Algorithm B",
        "Paired observations",
        "Median A",
        "Median B",
        "Median paired difference A-B",
        "Raw p-value",
        adjusted_column,
        "Statistically significant",
        effect_column,
        "Absolute effect size",
        "Effect magnitude",
        "Preferred algorithm (lower metric)",
    ]
    existing = [column for column in columns if column in frame.columns]

    return (
        frame[existing]
        .sort_values(
            ["Group", "Metric", "Absolute effect size"],
            ascending=[True, True, False],
        )
        .reset_index(drop=True)
    )


def build_friedman_effect_size_table(
    friedman_results: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Build a publication-ready Friedman/Kendall-W effect-size table."""
    if friedman_results is None or friedman_results.empty:
        return pd.DataFrame()

    frame = friedman_results.copy()
    if "Kendall W" not in frame.columns:
        raise KeyError("The Friedman table has no 'Kendall W' column.")

    frame["Kendall W magnitude"] = frame["Kendall W"].apply(
        classify_kendall_w
    )
    frame["Statistically significant"] = (
        pd.to_numeric(frame["p-value"], errors="coerce") < alpha
    )

    columns = [
        "Group",
        "Metric",
        "Complete blocks",
        "Algorithms",
        "Friedman chi-square",
        "Degrees of freedom",
        "p-value",
        "Statistically significant",
        "Kendall W",
        "Kendall W magnitude",
        "Note",
    ]
    existing = [column for column in columns if column in frame.columns]
    return frame[existing].reset_index(drop=True)


def nemenyi_critical_difference(
    number_of_algorithms: int,
    number_of_blocks: int,
    *,
    alpha: float = 0.05,
) -> float:
    """
    Compute the Nemenyi critical difference.

    CD = q_alpha * sqrt(k(k+1)/(6N)),
    where q_alpha is the Studentized-range critical value divided by sqrt(2).
    """
    if number_of_algorithms < 2 or number_of_blocks < 2:
        return np.nan

    q_alpha = (
        studentized_range.ppf(
            1.0 - alpha,
            number_of_algorithms,
            np.inf,
        )
        / math.sqrt(2.0)
    )
    return float(
        q_alpha
        * math.sqrt(
            number_of_algorithms
            * (number_of_algorithms + 1)
            / (6.0 * number_of_blocks)
        )
    )


def build_cd_summary_table(
    average_ranks: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Add the Nemenyi critical difference to every average-rank table."""
    if average_ranks is None or average_ranks.empty:
        return pd.DataFrame()

    required = {
        "Group",
        "Metric",
        "Algorithm",
        "Average rank",
        "Complete blocks",
    }
    missing = required - set(average_ranks.columns)
    if missing:
        raise KeyError(
            "Cannot build CD summary; missing columns: "
            f"{sorted(missing)}"
        )

    frames: List[pd.DataFrame] = []

    for (group, metric), subset in average_ranks.groupby(
        ["Group", "Metric"],
        observed=True,
        sort=False,
    ):
        subset = subset.copy()
        algorithm_count = int(subset["Algorithm"].nunique())
        block_count = int(
            pd.to_numeric(
                subset["Complete blocks"],
                errors="coerce",
            ).dropna().iloc[0]
        )
        critical_difference = nemenyi_critical_difference(
            algorithm_count,
            block_count,
            alpha=alpha,
        )
        subset["Algorithms"] = algorithm_count
        subset["Critical difference"] = critical_difference
        subset["Alpha"] = alpha
        frames.append(subset)

    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )


def _sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return cleaned.strip("_") or "output"


def _maximal_nonsignificant_intervals(
    ranks: np.ndarray,
    critical_difference: float,
) -> List[Tuple[int, int]]:
    """
    Find maximal contiguous sets whose extreme average ranks differ by no
    more than the Nemenyi critical difference.
    """
    intervals: List[Tuple[int, int]] = []
    count = len(ranks)

    for start in range(count):
        farthest = start
        for end in range(start + 1, count):
            if ranks[end] - ranks[start] <= critical_difference + 1e-12:
                farthest = end
            else:
                break
        if farthest > start:
            intervals.append((start, farthest))

    maximal: List[Tuple[int, int]] = []
    for interval in intervals:
        if not any(
            other != interval
            and other[0] <= interval[0]
            and other[1] >= interval[1]
            for other in intervals
        ):
            maximal.append(interval)

    return maximal


def save_critical_difference_diagram(
    rank_table: pd.DataFrame,
    *,
    group: str,
    metric: str,
    alpha: float = 0.05,
    output_dir: Optional[Path] = None,
    show: bool = False,
) -> Dict[str, object]:
    """
    Save a Demsar-style critical-difference diagram as PNG, PDF, and SVG.

    Lower average rank is better. Algorithms joined by a horizontal segment
    are not separated by more than the Nemenyi critical difference.
    """
    subset = rank_table[
        (rank_table["Group"].astype(str) == str(group))
        & (rank_table["Metric"].astype(str) == str(metric))
    ].copy()

    subset["Average rank"] = pd.to_numeric(
        subset["Average rank"],
        errors="coerce",
    )
    subset = (
        subset.dropna(subset=["Average rank"])
        .sort_values("Average rank")
        .reset_index(drop=True)
    )

    if len(subset) < 2:
        return {
            "Group": group,
            "Metric": metric,
            "Status": "Skipped: fewer than two algorithms",
        }

    block_count = int(
        pd.to_numeric(
            subset["Complete blocks"],
            errors="coerce",
        ).dropna().iloc[0]
    )
    algorithm_count = int(subset["Algorithm"].nunique())
    critical_difference = nemenyi_critical_difference(
        algorithm_count,
        block_count,
        alpha=alpha,
    )

    ranks = subset["Average rank"].to_numpy(dtype=float)
    algorithms = subset["Algorithm"].astype(str).tolist()
    intervals = _maximal_nonsignificant_intervals(
        ranks,
        critical_difference,
    )

    figure_width = max(10.0, 1.35 * algorithm_count + 4.5)
    figure_height = max(4.8, 0.48 * algorithm_count + 2.8)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    minimum_rank = 1.0
    maximum_rank = float(algorithm_count)
    axis_y = 1.0

    ax.hlines(
        axis_y,
        minimum_rank,
        maximum_rank,
        linewidth=1.5,
    )

    ticks = np.arange(1, algorithm_count + 1)
    for tick in ticks:
        ax.vlines(tick, axis_y - 0.07, axis_y + 0.07, linewidth=1.0)
        ax.text(
            tick,
            axis_y + 0.13,
            f"{tick:g}",
            ha="center",
            va="bottom",
        )

    split = int(math.ceil(algorithm_count / 2))
    left_indices = list(range(split))
    right_indices = list(range(split, algorithm_count))

    left_label_x = minimum_rank - 0.12
    right_label_x = maximum_rank + 0.12

    for row_index, index in enumerate(left_indices):
        y = axis_y - 0.42 - 0.34 * row_index
        rank = ranks[index]
        ax.plot([left_label_x, rank], [y, y], linewidth=1.0)
        ax.plot([rank, rank], [y, axis_y], linewidth=1.0)
        ax.scatter([rank], [axis_y], s=22, zorder=3)
        ax.text(
            left_label_x - 0.04,
            y,
            f"{algorithms[index]}  ({rank:.3f})",
            ha="right",
            va="center",
        )

    for row_index, index in enumerate(right_indices):
        y = axis_y - 0.42 - 0.34 * row_index
        rank = ranks[index]
        ax.plot([rank, right_label_x], [y, y], linewidth=1.0)
        ax.plot([rank, rank], [y, axis_y], linewidth=1.0)
        ax.scatter([rank], [axis_y], s=22, zorder=3)
        ax.text(
            right_label_x + 0.04,
            y,
            f"({rank:.3f})  {algorithms[index]}",
            ha="left",
            va="center",
        )

    connection_base_y = (
        axis_y
        - 0.42
        - 0.34 * max(len(left_indices), len(right_indices))
        - 0.12
    )
    for level, (start, end) in enumerate(intervals):
        y = connection_base_y - 0.17 * level
        ax.hlines(
            y,
            ranks[start],
            ranks[end],
            linewidth=4.0,
        )

    cd_y = axis_y + 0.52
    cd_start = maximum_rank - critical_difference
    cd_end = maximum_rank
    if cd_start < minimum_rank:
        cd_start = minimum_rank
    ax.hlines(cd_y, cd_start, cd_end, linewidth=2.0)
    ax.vlines(
        [cd_start, cd_end],
        cd_y - 0.06,
        cd_y + 0.06,
        linewidth=1.5,
    )
    ax.text(
        (cd_start + cd_end) / 2.0,
        cd_y + 0.10,
        f"CD = {critical_difference:.3f}",
        ha="center",
        va="bottom",
    )

    ax.text(
        minimum_rank,
        cd_y + 0.24,
        "Better",
        ha="left",
        va="bottom",
    )
    ax.annotate(
        "",
        xy=(minimum_rank, cd_y + 0.20),
        xytext=(minimum_rank + 0.65, cd_y + 0.20),
        arrowprops={"arrowstyle": "->"},
    )

    title = (
        f"Critical Difference Diagram — {metric}\n"
        f"{group}; N={block_count}, alpha={alpha:g}"
    )
    ax.set_title(title)
    ax.set_xlim(minimum_rank - 2.8, maximum_rank + 2.8)
    bottom = (
        connection_base_y
        - 0.17 * max(len(intervals), 1)
        - 0.30
    )
    ax.set_ylim(bottom, cd_y + 0.55)
    ax.axis("off")
    fig.tight_layout()

    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else RESULTS_DIR / "statistics" / "cd_diagrams"
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    stem = _sanitize_filename(f"{group}__{metric}__CD")
    paths = {
        "PNG": dataset_output_path(target_dir / f"{stem}.png"),
        "PDF": dataset_output_path(target_dir / f"{stem}.pdf"),
#         "SVG": target_dir / f"{stem}.svg",
    }
    fig.savefig(paths["PNG"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["PDF"], bbox_inches="tight")

    if CONFIG.get("save_plot_data_csv", True):
        cd_plot_data = subset[
            ["Group", "Metric", "Algorithm", "Average rank", "Complete blocks"]
        ].copy()
        cd_plot_data["Critical difference"] = critical_difference
        cd_plot_data["Alpha"] = alpha
        cd_data_path = save_plot_data_csv(
            cd_plot_data,
            paths["PNG"],
            label="plot_data",
        )
#     fig.savefig(paths["SVG"], bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return {
        "Group": group,
        "Metric": metric,
        "Complete blocks": block_count,
        "Algorithms": algorithm_count,
        "Critical difference": critical_difference,
        "Non-significant bars": len(intervals),
        "PNG": str(paths["PNG"]),
        "PDF": str(paths["PDF"]),
#         "SVG": str(paths["SVG"]),
        "Status": "Saved",
    }


def generate_effect_size_and_cd_outputs(
    statistical_outputs: Dict[str, pd.DataFrame],
    *,
    alpha: float = 0.05,
    output_dir: Optional[Path] = None,
    show_diagrams: bool = False,
) -> Dict[str, object]:
    """
    Generate effect-size tables and CD diagrams from run_statistical_analysis().
    """
    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else RESULTS_DIR / "statistics"
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    wilcoxon_effects = build_wilcoxon_effect_size_table(
        statistical_outputs.get("wilcoxon", pd.DataFrame()),
        alpha=alpha,
    )
    friedman_effects = build_friedman_effect_size_table(
        statistical_outputs.get("friedman", pd.DataFrame()),
        alpha=alpha,
    )
    cd_summary = build_cd_summary_table(
        statistical_outputs.get("average_ranks", pd.DataFrame()),
        alpha=alpha,
    )

    wilcoxon_path = dataset_output_path(target_dir / "wilcoxon_effect_sizes.csv")
    friedman_path = dataset_output_path(target_dir / "friedman_kendall_w_effect_sizes.csv")
    cd_summary_path = dataset_output_path(target_dir / "critical_difference_average_ranks.csv")

    wilcoxon_effects.to_csv(wilcoxon_path, index=False)
    friedman_effects.to_csv(friedman_path, index=False)
    cd_summary.to_csv(cd_summary_path, index=False)

    print(f"Saved {wilcoxon_path.resolve()}")
    print(f"Saved {friedman_path.resolve()}")
    print(f"Saved {cd_summary_path.resolve()}")

    diagram_rows: List[Dict[str, object]] = []
    average_ranks = statistical_outputs.get(
        "average_ranks",
        pd.DataFrame(),
    )

    if not average_ranks.empty:
        group_metric_pairs = (
            average_ranks[["Group", "Metric"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        for group, metric in group_metric_pairs:
            diagram_rows.append(
                save_critical_difference_diagram(
                    average_ranks,
                    group=str(group),
                    metric=str(metric),
                    alpha=alpha,
                    output_dir=target_dir / "cd_diagrams",
                    show=show_diagrams,
                )
            )

    diagram_index = pd.DataFrame(diagram_rows)
    diagram_index_path = dataset_output_path(target_dir / "critical_difference_diagram_index.csv")
    diagram_index.to_csv(diagram_index_path, index=False)
    print(f"Saved {diagram_index_path.resolve()}")

    return {
        "wilcoxon_effect_sizes": wilcoxon_effects,
        "friedman_effect_sizes": friedman_effects,
        "cd_average_ranks": cd_summary,
        "cd_diagram_index": diagram_index,
    }


def display_effect_size_summary(
    enhanced_statistics: Dict[str, object],
) -> None:
    """Display the main publication-ready statistical tables."""
    print("Wilcoxon rank-biserial effect sizes")
    display(enhanced_statistics["wilcoxon_effect_sizes"])

    print("Friedman effect sizes: Kendall's W")
    display(enhanced_statistics["friedman_effect_sizes"])

    print("Average ranks and Nemenyi critical differences")
    display(enhanced_statistics["cd_average_ranks"])

    print("Saved Critical Difference diagrams")
    display(enhanced_statistics["cd_diagram_index"])


# In[26]:



def build_per_map_summary(
    per_run: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate benchmark metrics separately for every map and algorithm.

    Mean, median, and sample standard deviation are stored for every
    available numeric evaluation criterion.
    """
    if per_run.empty:
        return pd.DataFrame()

    successful = per_run[
        per_run["Success"].astype(bool)
    ].copy()

    group_columns = [
        "Map name",
        "Algorithm",
    ]

    base = per_run.groupby(
        group_columns,
        observed=False,
    ).agg(
        Runs=("Success", "size"),
        Successful_runs=("Success", "sum"),
        Success_rate=("Success", "mean"),
    )

    metric_columns = [
        ("path_length", "Path length"),
        ("dataset_reference_gap_percent", "Optimality gap percent"),
        ("gap_vs_Astar_percent", "Gap vs A* percent"),
        ("time_ms", "Time ms"),
        ("expanded_nodes", "Expanded nodes"),
        ("generated_successors", "Generated successors"),
        ("queue_pushes", "Queue pushes"),
        ("jump_calls", "Jump calls"),
        ("turns", "Turns"),
        ("total_turning_angle_deg", "Total turning angle deg"),
        ("waypoints", "Waypoints"),
        ("search_objective", "Search objective"),
        ("turn_severity_45deg_units", "Turn severity 45deg units"),
    ]

    named_aggregations = {}

    for output_stem, source_column in metric_columns:
        if source_column not in successful.columns:
            continue

        named_aggregations[
            f"Mean_{output_stem}"
        ] = (source_column, "mean")

        named_aggregations[
            f"Median_{output_stem}"
        ] = (source_column, "median")

        named_aggregations[
            f"Std_{output_stem}"
        ] = (source_column, "std")

        named_aggregations[
            f"Min_{output_stem}"
        ] = (source_column, "min")

        named_aggregations[
            f"Max_{output_stem}"
        ] = (source_column, "max")

    numeric = (
        successful
        .groupby(
            group_columns,
            observed=False,
        )
        .agg(**named_aggregations)
    )

    summary = base.join(
        numeric,
        how="left",
    ).reset_index()

    summary["Success_rate"] *= 100.0

    summary["Algorithm"] = pd.Categorical(
        summary["Algorithm"],
        categories=ALGORITHM_ORDER,
        ordered=True,
    )

    return (
        summary
        .sort_values(
            ["Map name", "Algorithm"]
        )
        .reset_index(drop=True)
    )

def save_per_map_metric_plot(
    per_map_summary: pd.DataFrame,
    metric: str,
    title: str,
    filename: str,
    ylabel: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Plot one metric across maps, using one line for each algorithm."""
    if metric not in per_map_summary.columns:
        print(f"Skipping per-map plot; missing column: {metric}")
        return

    plot_columns = [
        "Map name",
        "Algorithm",
        metric,
    ]

    std_column = corresponding_std_column(
        metric
    )

    if (
        std_column is not None
        and std_column in per_map_summary.columns
    ):
        plot_columns.append(std_column)

    plot_frame = per_map_summary[
        plot_columns
    ].copy()
    plot_frame[metric] = pd.to_numeric(plot_frame[metric], errors='coerce')
    plot_frame = plot_frame.dropna(subset=[metric])

    if plot_frame.empty:
        print(f"Skipping per-map plot; no valid values for: {metric}")
        return

    map_names = sorted(plot_frame['Map name'].astype(str).unique())
    x_positions = np.arange(len(map_names))

    figure_width = max(12.0, min(32.0, 0.55 * len(map_names) + 7.0))
    fig, ax = plt.subplots(figsize=(figure_width, 6.5))

    for algorithm in ALGORITHM_ORDER:
        algorithm_frame = plot_frame[
            plot_frame['Algorithm'].astype(str) == algorithm
        ].copy()
        if algorithm_frame.empty:
            continue

        series = (
            algorithm_frame
            .set_index('Map name')[metric]
            .reindex(map_names)
        )

        ax.plot(
            x_positions,
            series.to_numpy(dtype=float),
            marker='o',
            linewidth=1.8,
            markersize=4,
            label=algorithm,
        )

    ax.set_title(title)
    ax.set_xlabel('Map')
    ax.set_ylabel(ylabel or metric.replace('_', ' '))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(map_names, rotation=60, ha='right')
    ax.grid(axis='both', alpha=0.25)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()

    target_dir = Path(output_dir) if output_dir is not None else RESULTS_DIR / 'per_map_plots'
    target_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_output_path(target_dir / filename)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    if CONFIG.get('save_plot_data_csv', True):
        data_path = save_plot_data_csv(plot_frame, path)
        print(f'Saved plot data: {data_path.resolve()}')

    if CONFIG.get('show_per_map_plots', True):
        plt.show()
    plt.close(fig)
    print(f"Saved {path.resolve()}")


def save_all_per_map_plots(per_map_summary: pd.DataFrame) -> None:
    """Create one graph per metric across all selected maps."""
    plot_specs = [
        ('Median_time_ms', 'Median runtime per map', 'median_runtime_per_map.png', 'Median runtime (ms)'),
        ('Mean_time_ms', 'Mean runtime per map', 'mean_runtime_per_map.png', 'Mean runtime (ms)'),
        ('Median_expanded_nodes', 'Median expanded nodes per map', 'median_expanded_nodes_per_map.png', 'Median expanded nodes'),
        ('Mean_expanded_nodes', 'Mean expanded nodes per map', 'mean_expanded_nodes_per_map.png', 'Mean expanded nodes'),
        ('Median_generated_successors', 'Median generated successors per map', 'median_generated_successors_per_map.png', 'Median generated successors'),
        ('Median_path_length', 'Median path length per map', 'median_path_length_per_map.png', 'Median path length'),
        ('Mean_gap_vs_Astar_percent', 'Mean path-length gap versus A* per map', 'mean_gap_vs_astar_per_map.png', 'Mean gap versus A* (%)'),
        ('Median_turns', 'Median number of turns per map', 'median_turns_per_map.png', 'Median turns'),
        ('Mean_turns', 'Mean number of turns per map', 'mean_turns_per_map.png', 'Mean turns'),
        ('Median_total_turning_angle_deg', 'Median total turning angle per map', 'median_turning_angle_per_map.png', 'Median total turning angle (degrees)'),
        ('Median_waypoints', 'Median waypoints per map', 'median_waypoints_per_map.png', 'Median waypoints'),
        ('Success_rate', 'Success rate per map', 'success_rate_per_map.png', 'Success rate (%)'),
    ]

    output_dir = RESULTS_DIR / 'per_map_plots'
    for metric, title, filename, ylabel in plot_specs:
        save_per_map_metric_plot(
            per_map_summary=per_map_summary,
            metric=metric,
            title=title,
            filename=filename,
            ylabel=ylabel,
            output_dir=output_dir,
        )


def save_standard_plots(summary_df: pd.DataFrame) -> None:
    """Save the standard sorted benchmark bar plots."""
    plot_specs = [
        ('Median_time_ms', 'Median runtime', 'median_runtime.png', True),
        ('Median_expanded_nodes', 'Median expanded nodes',
         'median_expanded_nodes.png', True),
        ('Median_turns', 'Median number of turns',
         'median_turns.png', True),
        ('Mean_gap_vs_Astar_percent', 'Mean path-length gap versus A*',
         'mean_gap_vs_astar.png', True),
    ]

    for metric, title, filename, ascending in plot_specs:
        if metric not in summary_df.columns:
            print(f"Skipping plot; missing column: {metric}")
            continue
        save_bar_plot(
            summary_df,
            'Algorithm',
            metric,
            title,
            filename,
            ascending=ascending,
        )


def run_experiment(config: Dict[str, object]) -> Dict[str, object]:
    """
    Execute the complete experiment according to CONFIG.

    Returns a dictionary containing generated DataFrames, example runs,
    selected maps, and optional archive path.
    """
    global CONFIG, selected_maps, RESULTS_DIR, MAPS_DIR, SCENARIOS_DIR
    
    

    CONFIG = dict(config)
    np.random.seed(int(CONFIG.get("random_seed", 42)))

    RESULTS_DIR = Path(CONFIG["results"])
    MAPS_DIR = Path(CONFIG["maps_dir"])
    SCENARIOS_DIR = Path(CONFIG["scenarios_dir"])

    for directory in (
        RESULTS_DIR,
        MAPS_DIR,
        SCENARIOS_DIR,
        Path(CONFIG["benchmark_cache_dir"]),
    ):
        directory.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 72)
    print("RUNNING EXPERIMENT")
    print("=" * 72)
    print(f"Dataset       : {CONFIG['dataset_name']}")
    print(f"Configuration : {CONFIG.get('config_name', '')}")
    print(f"Results       : {RESULTS_DIR.resolve()}")
    print("=" * 72)

    if CONFIG.get("download_data", True):
        download_and_extract(
            str(CONFIG["maps_url"]),
            MAPS_DIR,
        )
        download_and_extract(
            str(CONFIG["scenarios_url"]),
            SCENARIOS_DIR,
        )

    selected_maps = discover_selected_maps()
#     print(selected_maps[:0:5])
    if not selected_maps:
        raise FileNotFoundError(
            "No matching maps were found. Check map_sizes, map_variant, "
            "map_limit, and the downloaded dataset."
        )

    print(f"Selected {len(selected_maps)} map(s):")
    for map_path in selected_maps:
        print(f"  {map_path.name}")

    outputs: Dict[str, object] = {
        'selected_maps': selected_maps,
    }

    if CONFIG.get('run_smoke_test', True):
        print("\nRunning synthetic smoke test...")
        smoke_df = run_synthetic_smoke_test()
        outputs['smoke_test'] = smoke_df
        display(smoke_df)

    if not CONFIG.get('run_main_benchmark', True):
        print("\nMain benchmark disabled in CONFIG.")
        return outputs

    print("\nRunning dataset benchmark...")
    per_run_df, example_runs = benchmark_dataset(
        approaches=CONFIG.get('benchmark_approaches'),
        use_cache=bool(CONFIG.get('use_benchmark_cache', True)),
        force_recompute=bool(CONFIG.get('force_recompute_benchmark', False)),
    )
    per_run_df = reuse_astar_references(per_run_df, CONFIG)
    per_run_path = dataset_output_path(RESULTS_DIR / 'per_run_results.csv')
    per_run_df.to_csv(per_run_path, index=False)
    outputs['per_run'] = per_run_df
    outputs['example_runs'] = example_runs
    print(f"Saved {per_run_path.resolve()}")

    # ----------------------------------------------------------
    # Map structural descriptors and map-type summaries
    # ----------------------------------------------------------
    map_feature_df = (
        per_run_df[
            [
                "Map name",
                "Map type",
                "Map width",
                "Map height",
                "Free-cell ratio",
                "Obstacle density",
                "Dead-end ratio",
                "Corridor ratio",
                "Junction ratio",
                "Obstacle-edge density",
                "Mean axis visibility",
                "Connected components",
                "Largest component ratio",
            ]
        ]
        .drop_duplicates(subset=["Map name"])
        .sort_values("Map name")
        .reset_index(drop=True)
    )

    if CONFIG.get("run_structural_clustering", True):
        map_feature_df = add_structural_clusters(
            map_feature_df,
            number_of_clusters=int(
                CONFIG.get("map_cluster_count", 4)
            ),
        )

    map_features_path = dataset_output_path(RESULTS_DIR / "map_structure_features.csv")
    map_feature_df.to_csv(map_features_path, index=False)
    outputs["map_features"] = map_feature_df
    print(f"Saved {map_features_path.resolve()}")

    if "Structural cluster" in map_feature_df.columns:
        cluster_profile_columns = [
            column
            for column in MAP_FEATURE_COLUMNS
            if column in map_feature_df.columns
        ]
        cluster_profiles_df = (
            map_feature_df
            .dropna(subset=["Structural cluster"])
            .groupby("Structural cluster", observed=True)[
                cluster_profile_columns
            ]
            .mean()
            .reset_index()
        )
        cluster_profiles_path = (
            dataset_output_path(RESULTS_DIR / "map_structural_cluster_profiles.csv")
        )
        cluster_profiles_df.to_csv(
            cluster_profiles_path,
            index=False,
        )
        outputs["map_cluster_profiles"] = cluster_profiles_df
        print(f"Saved {cluster_profiles_path.resolve()}")

    summary_df = build_summary(per_run_df)
    summary_path = dataset_output_path(RESULTS_DIR / 'summary_results.csv')
    summary_df.to_csv(summary_path, index=False)
    outputs['summary'] = summary_df
    print(f"Saved {summary_path.resolve()}")

    per_map_summary_df = build_per_map_summary(per_run_df)
    per_map_summary_path = dataset_output_path(RESULTS_DIR / 'per_map_summary.csv')
    per_map_summary_df.to_csv(per_map_summary_path, index=False)
    outputs['per_map_summary'] = per_map_summary_df
    print(f"Saved {per_map_summary_path.resolve()}")

    if CONFIG.get('display_per_map_table', False):
        display(per_map_summary_df)

    if CONFIG.get('save_per_map_plots', True):
        save_all_per_map_plots(per_map_summary_df)

    map_type_summary_df = summarize_by_map_type(per_run_df)
    map_type_summary_path = (
        dataset_output_path(RESULTS_DIR / "algorithm_performance_by_map_type.csv")
    )
    map_type_summary_df.to_csv(
        map_type_summary_path,
        index=False,
    )
    outputs["map_type_summary"] = map_type_summary_df
    print(f"Saved {map_type_summary_path.resolve()}")

    map_type_rankings_df = create_map_type_rankings(
        map_type_summary_df
    )
    map_type_rankings_path = (
        dataset_output_path(RESULTS_DIR / "algorithm_rankings_by_map_type.csv")
    )
    map_type_rankings_df.to_csv(
        map_type_rankings_path,
        index=False,
    )
    outputs["map_type_rankings"] = map_type_rankings_df
#     print(f"Saved {map_type_rankings_path.resolve()}")

    paired_df = paired_vs_standard(per_run_df)
    paired_path = dataset_output_path(RESULTS_DIR / 'pairwise_vs_standard_jps.csv')
    paired_df.to_csv(paired_path, index=False)
    outputs['paired'] = paired_df
#     print(f"Saved {paired_path.resolve()}")

    if CONFIG.get("run_statistical_analysis", True):
        print("\nRunning non-parametric statistical analysis...")
        statistical_outputs = run_statistical_analysis(
            per_run_df,
            metrics=CONFIG.get(
                "statistical_metrics",
                DEFAULT_STATISTICAL_METRICS,
            ),
            algorithms=CONFIG.get(
                "statistical_algorithms",
                None,
            ),
            include_map_types=bool(
                CONFIG.get(
                    "statistics_by_map_type",
                    True,
                )
            ),
            alpha=float(
                CONFIG.get(
                    "statistical_alpha",
                    0.05,
                )
            ),
        )
        outputs["statistics"] = statistical_outputs

        if CONFIG.get("generate_effect_size_tables", True):
            print("\nGenerating effect-size tables and CD diagrams...")
            enhanced_statistics = generate_effect_size_and_cd_outputs(
                statistical_outputs,
                alpha=float(
                    CONFIG.get(
                        "statistical_alpha",
                        0.05,
                    )
                ),
                show_diagrams=bool(
                    CONFIG.get(
                        "show_cd_diagrams",
                        False,
                    )
                ),
            )
            outputs["enhanced_statistics"] = enhanced_statistics

            if CONFIG.get("display_effect_size_tables", True):
                display_effect_size_summary(enhanced_statistics)

    if CONFIG.get('display_tables', True):
        display(summary_df)
        display(map_type_summary_df)
        display(map_type_rankings_df)
        display(paired_df)

    if CONFIG.get('save_bar_plots', True):
        save_standard_plots(summary_df)

    if CONFIG.get('save_example_plot', True) and example_runs:
        example_name, example_data = next(iter(example_runs.items()))
        grid, start, goal, results = example_data
        print(f"Plotting example: {example_name}")
        plot_paths(
            grid=grid,
            start=start,
            goal=goal,
            results=results,
            filename=str(CONFIG.get(
                'example_plot_filename',
                'example_paths.png',
            )),
        )

    if CONFIG.get('run_sa_weight_sweep', False):
        print("\nRunning SA-JPS turn-weight sweep...")
        sa_sweep_df = benchmark_sa_weight_sweep()
        sa_sweep_path = dataset_output_path(RESULTS_DIR / 'sa_jps_weight_sweep_per_run.csv')
        sa_sweep_df.to_csv(sa_sweep_path, index=False)

        successful_sweep = sa_sweep_df[sa_sweep_df['Success']].copy()
        sa_sweep_summary_df = successful_sweep.groupby(
            'Turn penalty weight',
            as_index=False,
        ).agg(
            Runs=('Success', 'size'),
            Success_rate=('Success', 'mean'),
            Median_path_length=('Path length', 'median'),
            Mean_reference_gap_percent=(
                'Gap vs dataset reference percent', 'mean'
            ),
            Median_turns=('Turns', 'median'),
            Median_total_turning_angle_deg=(
                'Total turning angle deg', 'median'
            ),
            Median_waypoints=('Waypoints', 'median'),
            Median_expanded_nodes=('Expanded nodes', 'median'),
            Median_time_ms=('Time ms', 'median'),
            Median_search_objective=('Search objective', 'median'),
        )
        sa_sweep_summary_df['Success_rate'] *= 100.0

        sa_sweep_summary_path = (
            dataset_output_path(RESULTS_DIR / 'sa_jps_weight_sweep_summary.csv')
        )
        sa_sweep_summary_df.to_csv(
            sa_sweep_summary_path,
            index=False,
        )
        outputs['sa_weight_sweep'] = sa_sweep_df
        outputs['sa_weight_sweep_summary'] = sa_sweep_summary_df
#         print(f"Saved {sa_sweep_path.resolve()}")
#         print(f"Saved {sa_sweep_summary_path.resolve()}")

        if CONFIG.get('display_tables', True):
            display(sa_sweep_summary_df)

    if CONFIG.get('run_ablation_study', False):
        print("\nRunning complete ablation study...")
        ablation_outputs = run_ablation_study()
        outputs['ablation'] = ablation_outputs

    # Single-scenario inference is intentionally separate. Call get_path(...) after this function.

    if CONFIG.get('create_zip_archive', True):
        outputs['archive'] = package_results(
            RESULTS_DIR,
            str(CONFIG.get(
                'archive_name',
                'astar_jps_benchmark_results',
            )),
        )

    print("\nExperiment completed successfully.")
    return outputs


# ## 18. Inspect a specific map and scenario
# 
# Set the map filename and original scenario-row index in `CONFIG`. After the benchmark and ablation study, the notebook reruns all algorithms for that exact query, displays a metric table, and saves combined and individual path figures.
# 

# In[27]:


import json

# ==============================================================
# INDEPENDENT SINGLE-MAP / SINGLE-SCENARIO INFERENCE
# ==============================================================

def resolve_specific_map(map_name: Optional[str]) -> Path:
    """Resolve a map filename from selected or downloaded MovingAI maps."""
    candidates = list(selected_maps)

    if map_name is None or str(map_name).strip() == '':
        if not candidates:
            raise FileNotFoundError('No selected maps are available.')
        return candidates[0]

    requested = Path(str(map_name)).name
    for path in candidates:
        if path.name == requested:
            return path

    matches = list(MAPS_DIR.rglob(requested))
    if not matches:
        raise FileNotFoundError(
            f"Specific map {requested!r} was not found under {MAPS_DIR}."
        )
    return sorted(matches)[0]

def get_path(
    map_name: Optional[str],
    scenario_index: int,
    approaches: Optional[Sequence[str]] = None,
#     algorithm_parameters: Optional[
#         Dict[str, Dict[str, object]]
#     ] = None,
    *,
    save_figures: bool = True,
    show_figure: bool = True,
    output_name: Optional[str] = None,
) -> Dict[str, object]:
    """
    Apply selected path-planning approaches to one map/scenario.

    This function is independent from run_experiment(). It does not execute
    the complete benchmark or ablation study.

    Parameters
    ----------
    map_name:
        MovingAI map filename, for example 'Berlin_0_256.map'.
        None selects the first discovered map.

    scenario_index:
        Zero-based row index in the complete associated .scen file.

    approaches:
        Any subset of ALGORITHM_ORDER. None runs every approach.

    save_figures:
        Save one combined image and one image per successful approach.

    output_name:
        Optional output prefix. Defaults to map name + scenario index.

    Returns
    -------
    Dictionary containing metrics, SearchResult objects, and image paths.

    Example
    -------
    PATH_OUTPUT = get_path(
        map_name='Berlin_0_256.map',
        scenario_index=10,
        approaches=[
            'Standard JPS',
            'SA-JPS',
            'Fast Weighted SA-JPS',
        ],
    )
    """

    selected_approaches = approaches

    if selected_approaches is None:
        selected_approaches = list(ALGORITHM_ORDER)

    map_path = resolve_specific_map(map_name)
    grid = read_map(map_path)
    map_type, map_features = get_map_type(grid)

    scenario_path = locate_file(
        SCENARIOS_DIR,
        map_path.name + '.scen',
    )
    all_scenarios = read_scenario(scenario_path).dropna(
        subset=['start_x', 'start_y', 'goal_x', 'goal_y']
    ).reset_index(drop=True)

    scenario_index = int(scenario_index)
    if scenario_index < 0 or scenario_index >= len(all_scenarios):
        raise IndexError(
            f"scenario_index={scenario_index} is outside the valid range "
            f"0..{len(all_scenarios) - 1} for {map_path.name}."
        )

    scenario = all_scenarios.iloc[scenario_index]
    start = (int(scenario.start_x), int(scenario.start_y))
    goal = (int(scenario.goal_x), int(scenario.goal_y))
    reference_length = float(scenario.reference_length)

    print(
        f"Inference: map={map_path.name}, scenario={scenario_index}, "
        f"start={start}, goal={goal}, "
        f"map_type={map_type}, "
        f"approaches={selected_approaches}"
    )

    results = run_all_algorithms(
        grid,
        start,
        goal,
        approaches=selected_approaches,
    )



    rows = []
    astar_length = next(
        (
            float(result.metrics()['Path length'])
            for result in results
            if result.algorithm == 'A*' and result.success
        ),
        np.nan,
    )

    for result in results:
        metrics = result.metrics()
        measured_length = float(metrics['Path length'])
        gap_reference = (
            100.0 * (measured_length - reference_length) / reference_length
            if result.success and reference_length > 0
            else np.nan
        )
        gap_astar = (
            100.0 * (measured_length - astar_length) / astar_length
            if (
                result.success
                and np.isfinite(astar_length)
                and astar_length > 0
            )
            else np.nan
        )
        rows.append({
            'Map name': map_path.name,
            'Map type': map_type,
            'Obstacle density': map_features["Obstacle density"],
            'Corridor ratio': map_features["Corridor ratio"],
            'Dead-end ratio': map_features["Dead-end ratio"],
            'Junction ratio': map_features["Junction ratio"],
            'Mean axis visibility': map_features["Mean axis visibility"],
            'Obstacle-edge density': map_features["Obstacle-edge density"],
            'Largest component ratio': map_features["Largest component ratio"],
            'Scenario index': scenario_index,
            'Start x': start[0],
            'Start y': start[1],
            'Goal x': goal[0],
            'Goal y': goal[1],
            'Reference path length': reference_length,
            'Gap vs reference percent': gap_reference,
            'Gap vs A* percent': gap_astar,
            "Base algorithm": getattr(
                result,
                "base_algorithm",
                result.algorithm,
            ),

            "Configuration label": getattr(
                result,
                "configuration_label",
                result.algorithm,
            ),

            "Parameters": json.dumps(
                getattr(
                    result,
                    "parameters",
                    {},
                ),
                sort_keys=True,
            ),

            
            **metrics,
        })

    metrics_df = pd.DataFrame(rows)

    configuration_order = [
        result.algorithm
        for result in results
    ]

    metrics_df["Algorithm"] = pd.Categorical(
        metrics_df["Algorithm"],
        categories=configuration_order,
        ordered=True,
    )

    metrics_df = metrics_df.sort_values(
        "Algorithm"
    ).reset_index(drop=True)

    metrics_df["Algorithm"] = (
        metrics_df["Algorithm"]
        .astype(str)
    )

    output_dir = RESULTS_DIR / 'inference'
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = (
        safe_filename(output_name)
        if output_name
        else f"{safe_filename(map_path.stem)}_scenario_{scenario_index:05d}"
    )

    metrics_csv = dataset_output_path(output_dir / f'{prefix}_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
#     print(f'Saved metrics: {metrics_csv.resolve()}')
    display(metrics_df)

    image_outputs = {'combined': None, 'individual': {}}
    if save_figures:
        image_outputs = plot_paths(
            grid=grid,
            start=start,
            goal=goal,
            results=results,
            filename=f'inference/{prefix}_paths.png',
        )

    if not show_figure:
        # plot_paths currently saves and displays; this switch is retained
        # for API compatibility and can be connected to a plotting display
        # flag if a fully headless workflow is required.
        pass

    return {
        "map_path": map_path,
        "scenario_path": scenario_path,
        "scenario_index": scenario_index,
        "grid": grid,
        "start": start,
        "goal": goal,
        "reference_length": reference_length,
        "map_type": map_type,
        "map_features": map_features,
        "approaches": selected_approaches,
        "results": results,
        "metrics": metrics_df,
        "metrics_csv": metrics_csv,
        "combined_figure": image_outputs.get(
            "combined"
        ),
        "individual_figures": image_outputs.get(
            "individual",
            {},
        ),
    }


def inspect_specific_map_scenario(
    map_name: Optional[str],
    scenario_index: int,
    *,
    save_figures: bool = True,
    show_figure: bool = True,
) -> Dict[str, object]:
    """Backward-compatible wrapper around get_path()."""
    return get_path(
        map_name=map_name,
        scenario_index=scenario_index,
        approaches=None,
        save_figures=save_figures,
        show_figure=show_figure,
    )


# ## 19. Experiment configuration
# 

# In[28]:


# ==============================================================
# CONFIGURATION IS NOW STORED IN benchmark_config.py
# ==============================================================

# Jupyter:
#   Edit DEFAULT_DATASET and DEFAULT_CONFIG in the configuration selector cell.
#
# Python CLI after exporting the notebook to .py:
#   python benchmark.py street config1
#   python benchmark.py dao config2
#   python benchmark.py maze config3
#
# Available values:
# JSON requests execute precisely one experiment, not the notebook demo cells.
if ARGS.config_json is not None:
    # Restore the original server setting after IPython imports can lower it.
    sys.setrecursionlimit(30000)
    Path(CONFIG["results"]).mkdir(parents=True, exist_ok=True)
    (Path(CONFIG["results"]) / "effective_config.json").write_text(
        json.dumps(CONFIG, indent=2, default=str), encoding="utf-8"
    )
    EXPERIMENT_OUTPUTS = run_experiment(CONFIG)
    raise SystemExit(0)

print("Datasets:", sorted(DATASETS))
print("Profiles:", sorted(EXPERIMENT_CONFIGS))

display(
    pd.DataFrame(
        [
            {
                "Profile": name,
                "Description": values.get("description", ""),
                "Map limit": values.get("map_limit"),
                "Scenarios per map": values.get("scenarios_per_map"),
                "Ablation": values.get("run_ablation_study"),
            }
            for name, values in EXPERIMENT_CONFIGS.items()
        ]
    )
)

print("\nActive configuration:")
for key in (
    "dataset_name",
    "config_name",
    "maps_url",
    "scenarios_url",
    "maps_dir",
    "scenarios_dir",
    "results",
):
    print(f"{key:20s}: {CONFIG[key]}")


# In[ ]:


# Optional: switch configuration inside Jupyter without restarting.
#
#CONFIG = get_config(
#    dataset_name=CONFIG['dataset_name'],
#    profile_name="debug",
#    project_root=Path.cwd(),
#)

# Then rerun the path-initialization cell and call:
EXPERIMENT_OUTPUTS = run_experiment(CONFIG)


# In[ ]:


from benchmark_config import DATASETS

# print(DATASETS)
WANTED_DATASET1 = DATASETS.keys()
# WANTED_DATASET = ['ado']

tables = []
for ds in WANTED_DATASET1:
        dataset_config = get_config(
            dataset_name=ds,
            profile_name='full',
            project_root=Path.cwd(),
        )

        table = (
            generate_dataset_description_table(
                config=dataset_config,
                display_table=False,
                count_selected_scenarios = False
            )
        )

        tables.append(table)
        
combined = pd.concat(
    tables,
    ignore_index=True,
)
output = (
    Path.cwd()
    / "results"
    / "all_datasets_description.csv"
)

combined.to_csv(
    output,
    index=False,
)

display(combined)


# ## 20. Run the complete experiment
# 

# In[ ]:


# Run this cell after editing CONFIG.
EXPERIMENT_OUTPUTS = run_experiment(CONFIG)


# ## 21. Run inference separately with `get_path()`
# 
# This cell applies only the selected approaches to one specific map and scenario. It does not rerun the complete benchmark.
# 

# In[ ]:


approaches = [

    {
        "label": "Fast SA-JPS-Dijkstra r=4 tw=0.35 h=1.15",
        "algorithm": "Fast Weighted SA-JPS-Dijkstra Corridor",
        "parameters": {
            "corridor_radius": 4,
            "turn_weight": 0.35,
            "heuristic_weight": 1.15,
        },
    },


    {
        "label": "Bidirectional Standard JPS h=1.00",
        "algorithm": "Bidirectional Standard JPS",
        "parameters": {
            "heuristic_weight": 1.00,
        },
    },

    {
        "label": "Bidirectional SA-JPS tw=0.35",
        "algorithm": "Bidirectional SA-JPS",
        "parameters": {
            "turn_weight": 0.35,
            "heuristic_weight": 1.00,
        },
    },

    {
        "label": "Bidirectional Fast Weighted SA-JPS tw=0.35 h=1.50",
        "algorithm": "Bidirectional Fast Weighted SA-JPS",
        "parameters": {
            "turn_weight": 0.35,
            "heuristic_weight": 1.50,
        },
    },

    {
        "label": "A*",
        "algorithm": "A*",
        "parameters": {},
    },

    {
        "label": "Standard JPS h=1.00",
        "algorithm": "Standard JPS",
        "parameters": {
            "heuristic_weight": 1.00,
        },
    },
    {
        "label": "Standard JPS h=1.20",
        "algorithm": "Standard JPS",
        "parameters": {
            "heuristic_weight": 1.20,
        },
    },

    {
        "label": "Smooth JPS h=1.00",
        "algorithm": "Smooth JPS",
        "parameters": {
            "heuristic_weight": 1.00,
        },
    },
    {
        "label": "Smooth JPS h=1.20",
        "algorithm": "Smooth JPS",
        "parameters": {
            "heuristic_weight": 1.20,
        },
    },

    {
        "label": "SA-JPS tw=0.20",
        "algorithm": "SA-JPS",
        "parameters": {
            "turn_weight": 0.20,
            "heuristic_weight": 1.00,
        },
    },
    {
        "label": "SA-JPS tw=0.50",
        "algorithm": "SA-JPS",
        "parameters": {
            "turn_weight": 0.50,
            "heuristic_weight": 1.00,
        },
    },
    {
        "label": "SA-JPS tw=1.00",
        "algorithm": "SA-JPS",
        "parameters": {
            "turn_weight": 1.00,
            "heuristic_weight": 1.00,
        },
    },

    {
        "label": "Fast SA h=1.10 tw=0.35",
        "algorithm": "Fast Weighted SA-JPS",
        "parameters": {
            "turn_weight": 0.35,
            "heuristic_weight": 1.10,
        },
    },
    {
        "label": "Fast SA h=1.20 tw=0.35",
        "algorithm": "Fast Weighted SA-JPS",
        "parameters": {
            "turn_weight": 0.35,
            "heuristic_weight": 1.20,
        },
    },
    {
        "label": "Fast SA h=1.30 tw=0.35",
        "algorithm": "Fast Weighted SA-JPS",
        "parameters": {
            "turn_weight": 0.35,
            "heuristic_weight": 1.30,
        },
    },

    {
        "label": "Corridor r=2",
        "algorithm": "JPS-Dijkstra Corridor",
        "parameters": {
            "corridor_radius": 2,
        },
    },
    {
        "label": "Corridor r=4",
        "algorithm": "JPS-Dijkstra Corridor",
        "parameters": {
            "corridor_radius": 4,
        },
    },
    {
        "label": "Corridor r=8",
        "algorithm": "JPS-Dijkstra Corridor",
        "parameters": {
            "corridor_radius": 8,
        },
    },
]

PATH_OUTPUT = get_path(
    map_name="Berlin_0_1024.map",
    scenario_index=1000,
    approaches=approaches,
    output_name="all_parameter_configurations",
)

display(PATH_OUTPUT["metrics"])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




