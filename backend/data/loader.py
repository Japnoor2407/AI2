import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from backend.core.config import DATASET_DIR, PCU_WEIGHTS

logger = logging.getLogger(__name__)


class DatasetAdapter:
    """
    Robust adapter for Pune Heterogeneous CCTV Traffic Count Dataset.
    Extracts true arrivals from 25fps cumulative counter logs.
    """

    def __init__(self, dataset_root: Optional[Path] = None):
        self.root = dataset_root or DATASET_DIR
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_available_junctions(self) -> List[str]:
        if not self.root.exists():
            return []
        return sorted([d.name for d in self.root.iterdir() if d.is_dir()])

    def get_cameras_for_junction(self, junction: str) -> List[str]:
        j_dir = self.root / junction
        if not j_dir.exists():
            return []
        return sorted([d.name for d in j_dir.iterdir() if d.is_dir()])

    def get_dates_for_camera(self, junction: str, camera: str) -> List[str]:
        c_dir = self.root / junction / camera
        if not c_dir.exists():
            return []
        files = glob.glob(str(c_dir / f"{camera}_*.csv"))
        dates = set()
        for f in files:
            base = os.path.basename(f)
            # format e.g. a2_11.csv or a2_11_one.csv
            parts = base.replace(".csv", "").split("_")
            if len(parts) >= 2:
                dates.add(parts[1])
        return sorted(list(dates))

    def get_dataset_summary(self) -> Dict:
        junctions = self.get_available_junctions()
        summary = {
            "dataset_title": "Heterogeneous Traffic Count Dataset, Pune (Jan 2023)",
            "junction_count": len(junctions),
            "junctions": {},
            "total_files": 0,
            "vehicle_classes": ["car", "motorbike", "bus", "truck"],
            "pcu_standards": PCU_WEIGHTS,
        }
        for j in junctions:
            cameras = self.get_cameras_for_junction(j)
            j_info = {"cameras": {}, "camera_count": len(cameras)}
            for c in cameras:
                dates = self.get_dates_for_camera(j, c)
                c_dir = self.root / j / c
                c_files = glob.glob(str(c_dir / "*.csv"))
                summary["total_files"] += len(c_files)
                j_info["cameras"][c] = {
                    "file_count": len(c_files),
                    "recorded_dates": dates,
                }
            summary["junctions"][j] = j_info
        return summary

    def load_camera_arrivals(
        self,
        junction: str,
        camera: str,
        date: str = "11",
        max_seconds: Optional[int] = 1800,  # default 30 min window
    ) -> pd.DataFrame:
        """
        Loads camera raw CSVs, extracts second-by-second positive arrival increments.
        Returns a DataFrame indexed by second offset with columns:
        [time_str, direction, car, motorbike, bus, truck, pcu]
        """
        cache_key = f"{junction}_{camera}_{date}_{max_seconds}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        c_dir = self.root / junction / camera
        primary_file = c_dir / f"{camera}_{date}.csv"
        continuation_file = c_dir / f"{camera}_{date}_one.csv"

        if not primary_file.exists():
            raise FileNotFoundError(f"Primary file not found: {primary_file}")

        # Read primary file with chunking if needed or direct if sized reasonably
        # 1800 seconds * 100 rows/sec = ~180,000 rows
        nrows = None
        if max_seconds:
            # Buffer for 100 frames/sec plus safety margin
            nrows = (max_seconds + 60) * 100

        df_raw = pd.read_csv(primary_file, nrows=nrows)
        df_raw = df_raw.dropna(subset=["Time", "Direction"])

        # Coerce counts to non-negative numeric
        for col in ["car", "motorbike", "bus", "truck"]:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce").fillna(0).astype(int)
            else:
                df_raw[col] = 0

        # Collapse 25 fps frames into 1-second max cumulative values per direction
        df_sec = df_raw.groupby(["Time", "Direction"], as_index=False)[["car", "motorbike", "bus", "truck"]].max()

        # Sort chronologically by Time
        df_sec = df_sec.sort_values(by=["Direction", "Time"]).reset_index(drop=True)

        # Compute arrival difference: delta = count(t) - count(t-1)
        arrival_records = []
        for direction, group in df_sec.groupby("Direction"):
            g = group.copy()
            for col in ["car", "motorbike", "bus", "truck"]:
                diff = g[col].diff().fillna(0)
                # Filter out negative diffs (e.g. from session counter resets)
                diff = diff.clip(lower=0)
                g[f"{col}_arrivals"] = diff.astype(int)

            # Compute PCU for arrivals
            g["pcu_arrivals"] = (
                g["car_arrivals"] * PCU_WEIGHTS["car"]
                + g["motorbike_arrivals"] * PCU_WEIGHTS["motorbike"]
                + g["bus_arrivals"] * PCU_WEIGHTS["bus"]
                + g["truck_arrivals"] * PCU_WEIGHTS["truck"]
            )
            arrival_records.append(g)

        if not arrival_records:
            return pd.DataFrame()

        df_arrivals = pd.concat(arrival_records, ignore_index=True)
        # Order by Time ascending
        df_arrivals = df_arrivals.sort_values(by=["Time", "Direction"]).reset_index(drop=True)

        if max_seconds:
            unique_times = df_arrivals["Time"].drop_duplicates().head(max_seconds)
            df_arrivals = df_arrivals[df_arrivals["Time"].isin(unique_times)].reset_index(drop=True)

        self._cache[cache_key] = df_arrivals
        return df_arrivals.copy()

    def load_aggregated_intervals(
        self,
        junction: str,
        camera: str,
        date: str = "11",
        interval_sec: int = 600,
        max_intervals: int = 12,
    ) -> List[Dict]:
        """
        Aggregates traffic flow into consecutive N-minute intervals (default 10-min = 600s).
        Returns a list of structured 10-minute metrics for charting.
        """
        total_seconds = interval_sec * max_intervals
        df = self.load_camera_arrivals(junction, camera, date, max_seconds=total_seconds)
        if df.empty:
            return []

        unique_times = df["Time"].drop_duplicates().tolist()
        num_intervals = len(unique_times) // interval_sec
        if num_intervals == 0 and len(unique_times) > 0:
            num_intervals = 1

        results = []
        for i in range(min(num_intervals, max_intervals)):
            start_idx = i * interval_sec
            end_idx = min((i + 1) * interval_sec, len(unique_times))
            if start_idx >= len(unique_times):
                break
            bin_times = unique_times[start_idx:end_idx]
            sub_df = df[df["Time"].isin(bin_times)]

            car_sum = int(sub_df["car_arrivals"].sum())
            bike_sum = int(sub_df["motorbike_arrivals"].sum())
            bus_sum = int(sub_df["bus_arrivals"].sum())
            truck_sum = int(sub_df["truck_arrivals"].sum())
            pcu_sum = round(float(sub_df["pcu_arrivals"].sum()), 2)
            tot_veh = car_sum + bike_sum + bus_sum + truck_sum

            t_start = bin_times[0]
            t_end = bin_times[-1]
            label = f"{t_start[:5]}-{t_end[:5]}"

            # Scaled to hourly equivalent flow rate
            hourly_rate = round(pcu_sum * (3600.0 / len(bin_times)), 1) if bin_times else 0.0

            results.append({
                "interval_index": i + 1,
                "label": label,
                "start_time": t_start,
                "end_time": t_end,
                "duration_sec": len(bin_times),
                "car": car_sum,
                "motorbike": bike_sum,
                "bus": bus_sum,
                "truck": truck_sum,
                "total_vehicles": tot_veh,
                "pcu_arrivals": pcu_sum,
                "hourly_flow_pcu": hourly_rate,
            })

        return results


