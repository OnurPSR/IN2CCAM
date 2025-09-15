"""
LoadBalancing
-------------
Path enumeration and multi-objective path selection with optional
Sobol sensitivity analysis for (alpha, gamma, depth).

The planner operates on a directed graph defined by:
- `roads_trajectory`: mapping node -> list[(next_node, edge_length)]
- `duration_of_roads`: DataFrame with per-road duration breakpoints
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Optional SALib imports (deferred to keep SALib as a soft dependency)
# ---------------------------------------------------------------------
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
except ModuleNotFoundError as e:
    # Keep the exception and raise it only if Sobol analysis is used.
    saltelli = sobol = e


class roadPlanner(object):
    """
    Route planner that enumerates candidate paths up to a recursion limit
    and selects paths based on a convex combination of normalized duration
    and distance.

    Parameters
    ----------
    duration_of_roads : pd.DataFrame
        Table containing per-road duration breakpoints. Must include
        'description15' (road label) and columns used in `find_optimum`.
    roads_trajectory : dict[str, list[tuple[str, float]]]
        Adjacency mapping: node -> list of (next_node, edge_length).
    alpha : float, default 0.5
        Weight on normalized duration (1 - alpha on normalized distance).
    gamma : float, default 3.0
        Pruning factor: retains paths whose distance <= gamma * shortest_distance.
    seed : int | None, default None
        Seed for deterministic sampling.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        duration_of_roads,
        roads_trajectory,
        alpha: float = 0.5,
        gamma: float = 3.0,
        seed: int | None = None,
    ) -> None:
        self.duration_of_roads = duration_of_roads
        self.roads_trajectory = roads_trajectory
        self.alpha = float(alpha)
        self.gamma = float(gamma)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ------------------------------------------------------------------
    # Path enumeration (depth-limited DFS on `roads_trajectory`)
    # ------------------------------------------------------------------
    def shortest_path(self, source, destination, max_recursion, distance, path, roads_trajectory):
        """
        Enumerate candidate paths up to a maximum recursion depth.

        Parameters
        ----------
        source, destination : str
            Start and end node identifiers.
        max_recursion : int
            Remaining recursion depth (depth limit).
        distance : float
            Accumulated distance along the current partial path.
        path : list[str]
            Sequence of visited nodes/edges along the current path.
        roads_trajectory : dict
            Adjacency mapping used for traversal.

        Returns
        -------
        list[tuple[list[str], float]] | None
            List of (path_nodes, total_distance) tuples, or None if depth limit
            is reached without progress.
        """
        paths = []
        if max_recursion == 0:
            return None
        for key in roads_trajectory:
            if source in key:
                for nxt, length in roads_trajectory[key]:
                    temp_path = path + [key]
                    temp_dist = distance
                    if destination in nxt:
                        paths.append((temp_path + [nxt], temp_dist + length))
                    elif nxt in path:
                        continue
                    else:
                        sub = self.shortest_path(
                            nxt,
                            destination,
                            max_recursion - 1,
                            temp_dist + length,
                            temp_path + [nxt],
                            roads_trajectory,
                        )
                        if sub not in (None, []):
                            paths += sub if isinstance(sub, list) else [sub]
        return paths

    def to_row(self, paths):
        """
        Convert enumerated paths into (edge_labels, distance) tuples.

        Notes
        -----
        The planner expects road labels in the form 'FROM a TO' within
        'description15'; this function reconstructs that representation.
        """
        out, buf = [], []
        for path in paths:
            if len(path[0]) == 2:
                buf.append(f"{path[0][0]} a {path[0][1]}")
            else:
                for i in range(0, len(path[0]), 2):
                    buf.append(f"{path[0][i]} a {path[0][i+1]}")
            out.append((buf, path[1]))
            buf = []
        return out

    def find_optimum(self, paths, duration_of_roads, initial_value_temp):
        """
        Score candidate paths using a normalized weighted sum of duration and
        distance; then apply an updating rule to the reference column to
        simulate evolving conditions across chunks of size 100.

        Parameters
        ----------
        paths : list[tuple]
            Output from `shortest_path`.
        duration_of_roads : pd.DataFrame
            Duration table with columns: 'description15', [1..], and
            a 'reference column' added here.
        initial_value_temp : int
            Initial offset used to update the 'reference column' by chunks.

        Returns
        -------
        list[tuple]
            A list of best-scoring paths per chunk with elements:
            (edges, duration_score, total_distance, combined_score).
        """
        α, γ = self.alpha, self.gamma
        best_paths = []

        shortest = min(p[1] for p in paths)
        cand = [p for p in self.to_row(paths) if p[1] <= γ * shortest]
        if not cand:
            return best_paths

        dur_df = duration_of_roads.iloc[:, [1] + list(range(11, 31))].copy()
        dur_df["reference column"] = np.random.uniform(500, 1300, len(dur_df))

        # Chunk the update process into 100-unit steps (last chunk may be smaller).
        chunks = [100] * (initial_value_temp // 100)
        if (rem := initial_value_temp % 100):
            chunks.append(rem)

        for chunk in chunks:
            stats = []
            for edges, dist in cand:
                dur = 0
                for road in edges:
                    row = dur_df[dur_df["description15"] == road]
                    col, ref = 1, int(row["reference column"].iloc[0])
                    while True:
                        if col == 20:
                            dur += 21
                            break
                        start, end = int(row[col].iloc[0]), int(row[col + 1].iloc[0])
                        if start <= ref + chunk < end:
                            dur += col + 1
                            break
                        if ref + chunk < start:
                            dur += 1
                            break
                        col += 1
                stats.append((edges, dur, dist))

            durs = [d for _, d, _ in stats]
            dists = [s for _, _, s in stats]
            dmin, dmax, smin, smax = min(durs), max(durs), min(dists), max(dists)
            norm = lambda v, lo, hi: 0.0 if hi == lo else (v - lo) / (hi - lo)

            scored = [
                (edges, dur, dist, α * norm(dur, dmin, dmax) + (1 - α) * norm(dist, smin, smax))
                for edges, dur, dist in stats
            ]
            best = min(scored, key=lambda z: z[3])
            best_paths.append(best)

            # Update the 'reference column' for edges on the selected path.
            for road in best[0]:
                idx = dur_df[dur_df["description15"] == road].index[0]
                dur_df.loc[idx, "reference column"] += chunk

        return best_paths

    # ------------------------------------------------------------------
    def main(
        self,
        source: str,
        destination: str,
        initial_value_temp: int,
        recursion_depth: int = 20,
    ):
        """
        End-to-end planning pipeline: enumerate paths and select optima.
        """
        paths = self.shortest_path(
            source, destination, recursion_depth, 0, [], self.roads_trajectory
        )
        if paths == []:
            return "NO PATH AVALIABLE FOR THIS ROUTE"
        return self.find_optimum(paths, self.duration_of_roads, initial_value_temp)

    # ==================================================================
    #                         Sobol Sensitivity
    # ==================================================================
    def _evaluate_plan(
        self,
        source: str,
        destination: str,
        initial_value_temp: int,
        recursion_depth: int,
    ) -> float:
        """
        Compute average speed for the best path produced by `main`.
        Returns NaN if no path is found or duration is zero.
        """
        res = self.main(source, destination, initial_value_temp, recursion_depth)
        if isinstance(res, str):
            return np.nan

        total_dur = sum(r[1] for r in res)
        total_dist = sum(r[2] for r in res)
        return total_dist / total_dur if total_dur > 0 else np.nan

    @staticmethod
    def sobol_sensitivity(
        duration_of_roads,
        roads_trajectory,
        source: str,
        destination: str,
        initial_value_temp: int,
        depth_range: Tuple[int, int] = (10, 20),
        alpha_range: Tuple[float, float] = (0.3, 0.7),
        gamma_range: Tuple[float, float] = (2.0, 4.0),
        n_base: int = 512,
        seed: int | None = None,
        out_dir: str | Path = "results",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Estimate first-order Sobol indices for parameters (alpha, gamma, depth).

        Parameters
        ----------
        duration_of_roads, roads_trajectory : see class documentation
        source, destination : str
            Endpoints for planning.
        initial_value_temp : int
            Initial offset used in the duration update mechanism.
        depth_range : (int, int)
            Inclusive range for recursion depth (cast to int during sampling).
        alpha_range, gamma_range : (float, float)
            Parameter ranges for weighting and pruning.
        n_base : int
            Base sample size for Saltelli; total evaluations = n_base * (2k + 2).
        seed : int | None
            Random seed.
        out_dir : str | Path
            Directory to write 'sobol_summary.xlsx' and 'sobol_raw.csv'.

        Returns
        -------
        sobol_df : pd.DataFrame
            First-order Sobol indices (S1) with confidence intervals.
        raw_df : pd.DataFrame
            Sampled parameters and resulting average speeds.
        """
        if isinstance(sobol, ModuleNotFoundError):
            raise ModuleNotFoundError(
                "SALib not installed. Install with `pip install SALib` to enable Sobol analysis."
            )

        np.random.seed(seed)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        problem = {
            "num_vars": 3,
            "names": ["alpha", "gamma", "depth"],
            "bounds": [
                list(alpha_range),
                list(gamma_range),
                list(depth_range),
            ],
        }

        sample = saltelli.sample(problem, n_base, calc_second_order=False)
        outputs = []

        for α, γ, depth in sample:
            depth_int = int(round(depth))
            planner = roadPlanner(
                duration_of_roads, roads_trajectory, alpha=float(α), gamma=float(γ), seed=seed
            )
            speed = planner._evaluate_plan(
                source, destination, initial_value_temp, depth_int
            )
            outputs.append(speed)

        # Validate outputs before Sobol analysis.
        Y = np.asarray(outputs, dtype=float)
        if not np.isfinite(Y).all():
            raise ValueError(
                "Non-finite speeds encountered (NaN/inf). Verify parameter ranges or planner failures."
            )
        if Y.std() == 0:
            raise ValueError("Output variance is zero; Sobol indices are undefined.")

        # Build raw results table
        raw_df = pd.DataFrame(sample, columns=["alpha", "gamma", "depth"])
        raw_df["depth"] = raw_df["depth"].round().astype(int)
        raw_df["avg_speed"] = Y
        raw_csv = out_dir / "sobol_raw.csv"
        raw_df.to_csv(raw_csv, index=False)

        # First-order Sobol index estimation
        sob_res = sobol.analyze(problem, Y, calc_second_order=False)
        sobol_df = pd.DataFrame(
            {
                "parameter": problem["names"],
                "S1": sob_res["S1"],
                "S1_conf": sob_res["S1_conf"],
            }
        )

        # Write Excel summary + raw data
        excel_path = out_dir / "sobol_summary.xlsx"
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            sobol_df.to_excel(writer, sheet_name="S1", index=False)
            raw_df.to_excel(writer, sheet_name="raw_samples", index=False)

        return sobol_df, raw_df


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
duration_of_roads = pd.read_excel(
    "Talep-Hız-Yolculuk-Suresi.xlsx", sheet_name="Yolculuk Süresi-Hacim"
)
roads_trajectory = dict({})

for road, distance in zip(duration_of_roads['description15'], duration_of_roads['Distance']):
    fRom, to = tuple(road.split(' a '))
    if fRom not in roads_trajectory.keys():
        roads_trajectory[fRom] = []
        roads_trajectory[fRom].append((to, distance))
    else:
        roads_trajectory[fRom].append((to, distance))

# Sobol sensitivity analysis
sobol_df, raw_df = roadPlanner.sobol_sensitivity(
    duration_of_roads,
    roads_trajectory,
    source="FLORIDA-PADRE SEIXAS",
    destination="CASTRELOS-BALAIDOS",
    initial_value_temp=6075,
    depth_range=(10, 40),
    alpha_range=(0, 1.0),
    gamma_range=(1.5, 4.0),
    n_base=512,
    seed=42,
    out_dir="results",
)

print(sobol_df)
