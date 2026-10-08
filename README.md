# JPS benchmark: setup and experiments

This folder runs MovingAI grid-pathfinding experiments. The supplied JSON configurations run either the Weighted JPS baseline alone or a comparison with FW-SA-JPS. Both support resuming completed searches after interruption.

## 1. Environment and setup

Copy this entire folder to the server. Run the commands below from that directory. The current working directory determines where `datasets/` and result directories are located; use the same directory when restarting a job.

Create the supplied Conda environment:

```bash
conda env create -f environment.yml
conda activate astar_jps
export MPLBACKEND=Agg
```

Alternatively, for direct Python execution without Conda, use Python 3.10 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MPLBACKEND=Agg
```

The supplied Slurm script activates Conda. If using a virtual environment, replace its two Conda activation lines with `source .venv/bin/activate`.

Check the installation without running a benchmark:

```bash
python -m unittest test_weighted_resume.py
python run_benchmark.py dao --config-json exp_weighted_comparison.json --dry-run
```

The tests cover configuration dispatch, interruption/resume, duplicate prevention, reference matching, and journal identity. A dry run prints configuration only; it does not download data or execute searches.

## 2. Download datasets

Supported dataset names:

```text
bg512  bgmaps  dao  maze  random  sc1  street  wc3maps512
```

By default, the first experiment downloads and extracts the selected dataset's map and scenario archives using the URLs in `benchmark_config.py`. Internet access is required for this initial download.

Expected layout, relative to your working directory:

```text
datasets/
  dao/
    maps/          # .map files, optionally in subdirectories
    scenarios/     # matching <map-name>.map.scen files
```

If compute nodes cannot access the internet, download beforehand on a machine with network access. Generate a download-only configuration from the supplied comparison configuration:

```bash
python -c 'import json; from pathlib import Path; c=json.loads(Path("exp_weighted_comparison.json").read_text()); c["output_root"]="download_setup"; c["overrides"].update(download_data=True,run_main_benchmark=False,run_smoke_test=False); Path("exp_download.json").write_text(json.dumps(c,indent=2))'
python run_benchmark.py dao --config-json exp_download.json
```

This downloads the selected dataset without running searches. Repeat with other dataset names as needed, then copy `datasets/` to the benchmark working directory if downloaded elsewhere. Alternatively, manually download the map/scenario ZIPs listed in `benchmark_config.py` and extract them into the layout above.

The downloader skips any nonempty destination directory; it does not verify a partial download. If downloading or extraction was interrupted, check that the map/scenario files are complete before starting. For fully prepared data, you may set `"download_data": false` inside the experiment's `overrides`.

## 3. Select an experiment configuration

| JSON file | Methods | Parameters | Output root |
| --- | --- | --- | --- |
| `exp_config.json` | Weighted JPS only | No turn penalty, w = 1.15 | `results_weighted_jps` |
| `exp_weighted_comparison.json` | Weighted JPS and FW-SA-JPS | Both w = 1.15; FW-SA-JPS lambda = 0.35 | `results_weighted_comparison` |

Both configurations disable smoke tests, parameter sweeps, and statistical testing. They run the listed methods on identical selected scenarios and export descriptive results. Weighted JPS uses the Standard JPS implementation with heuristic weighting and no smoothness tie-break; it is not the SA-JPS implementation with its penalty disabled.

Configuration fields:

- `base_profile`: selects settings from `benchmark_config.py`. Both supplied files use `full`: all discovered maps and up to 200 uniformly selected scenarios per map.
- `output_root`: result destination relative to the working directory, or an absolute path.
- `overrides`: settings that replace the selected profile, including the explicit algorithm list and parameters.
- `astar_reference_root`: location of existing benchmark results used to obtain measured A* reference lengths; see Section 7.

**The JSON `base_profile` overrides the positional profile argument.** Passing `debug` while using a JSON with `"base_profile": "full"` still selects the full profile.

For a small initial check, create a separate configuration:

```bash
python -c 'import json; from pathlib import Path; c=json.loads(Path("exp_weighted_comparison.json").read_text()); c["output_root"]="results_weighted_comparison_small"; c["overrides"].update(map_limit=2,scenarios_per_map=5); Path("exp_comparison_small.json").write_text(json.dumps(c,indent=2))'
python run_benchmark.py dao --config-json exp_comparison_small.json
```

Choose workload limits before starting the final experiment. Changing the selected workload creates a different cache identity. To evaluate only one or a few datasets, simply run only those dataset names.

## 4. Run directly with Python

Compare Weighted JPS with FW-SA-JPS on one dataset:

```bash
python run_benchmark.py dao --config-json exp_weighted_comparison.json
```

Run the baseline alone:

```bash
python run_benchmark.py dao --config-json exp_config.json
```

Omitting `--config-json` selects `exp_config.json` beside the launcher. A custom configuration can be supplied using an absolute or relative path.

Run a chosen subset sequentially in Bash:

```bash
for dataset in dao street; do
    python run_benchmark.py "$dataset" --config-json exp_weighted_comparison.json || break
done
```

The underlying script also accepts the JSON directly:

```bash
python -u Astar_JPS_SA_JPS_config_profiles_ready.py dao full --config-json exp_weighted_comparison.json
```

Without a JSON, the underlying script uses the current `benchmark_config.py` profile, which can enable additional analyses and sweeps. Use the supplied JSON commands for the focused reviewer experiments.

## 5. Run with Slurm

Edit `run_job.slurm` for your cluster: partition, time limit, memory, CPU allocation, and Conda installation path. It currently requests the `mem` partition, four CPUs, 16 GB memory, and six days. The benchmark loop runs sequentially; allocating four CPUs does not launch four independent dataset jobs.

Create the log directory **before** submitting because Slurm opens log files before the script executes:

```bash
mkdir -p logs
export MPLBACKEND=Agg
sbatch run_job.slurm dao full exp_weighted_comparison.json
```

Arguments are: `dataset`, `profile`, `JSON file`. The third argument selects the experiment configuration. Omitting it uses `exp_config.json`, which runs only Weighted JPS.

Submit one job per chosen dataset:

```bash
for dataset in dao street; do
    sbatch run_job.slurm "$dataset" full exp_weighted_comparison.json
done
```

Inspect progress:

```bash
squeue -u "$USER"
tail -f logs/astar_job_JOBID.out
tail -f logs/astar_job_JOBID.err
```

Replace `JOBID` with the number returned by `sbatch`. Progress bars may appear in the error log; this alone does not indicate failure.

If Slurm reports DOS line breaks after a Windows transfer:

```bash
sed -i 's/\r$//' run_job.slurm
```

## 6. Resume after interruption

Repeat the same Python command or resubmit the same Slurm command from the same working directory. Keep the configuration, datasets, and output directory unchanged.

- Each completed method execution is committed to a SQLite journal under the result directory's `cache/` folder.
- Saved executions are skipped, including when one method completed before the other was interrupted in the same scenario.
- Only unfinished, uncommitted work may repeat. A completed benchmark loads its final cache.
- Compatible legacy pickle checkpoints are imported when the journal is empty. Unsaved work from older runs cannot be recovered.
- Keep `use_benchmark_cache=true` and `force_recompute_benchmark=false`. Changing input timestamps, selection, or algorithm parameters can produce a different cache identity.
- Do not launch concurrent jobs for the same dataset and output configuration.

Resume requires restarting or resubmitting the job; the code does not automatically requeue Slurm jobs. Map preparation and result export can repeat even when saved searches are skipped.

## 7. A* reference lengths and result files

The supplied experiments do not rerun A*. To report A* gaps, point `astar_reference_root` in the selected JSON to the original results root, for example:

```json
"astar_reference_root": "/path/to/original/results"
```

That root must contain:

```text
<dataset>/full/per_run_results_<dataset>.csv
```

Reference CSVs must contain measured `A* reference length` values and scenario identity columns. Matching uses map name, original scenario index, and start/goal coordinates. Missing references remain unavailable; scenario-file reference distances are not substituted. References are refreshed on cached runs too, so correcting the reference path does not require new searches.

Comparison outputs appear in:

```text
results_weighted_comparison/<dataset>/full/
  per_run_results_<dataset>.csv
  per_map_summary_<dataset>.csv
  summary_results_<dataset>.csv
  cache/
```

The baseline-only experiment uses `results_weighted_jps/<dataset>/full/`. Additional descriptive CSVs may also be generated.

Use the per-run or per-map files for paired comparisons on matching cases. The supplied configs do not automatically regenerate manuscript Tables I/II or perform the weighted-method significance comparison. Report the datasets and aggregation used; a subset experiment supports conclusions about that subset.
