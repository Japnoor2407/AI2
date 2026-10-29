"""
Multi-Object Vehicle Tracker & Virtual Tripwire Counting (Phase 4).
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from backend.vision.detector import VehicleDetection


class TrackedVehicle:
    def __init__(self, track_id: int, detection: VehicleDetection):
        self.track_id = track_id
        self.class_name = detection.class_name
        self.bbox = detection.bbox
        self.positions: List[Tuple[int, int]] = [detection.center]
        self.disappeared_frames = 0
        self.has_crossed = False


class CentroidTracker:
    """
    Associates vehicle detections across consecutive frames to form unique trajectories.
    """

    def __init__(self, max_disappeared: int = 15, max_distance: float = 60.0):
        self.next_track_id = 1
        self.objects: Dict[int, TrackedVehicle] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections: List[VehicleDetection]) -> List[TrackedVehicle]:
        if not detections:
            for obj in list(self.objects.values()):
                obj.disappeared_frames += 1
                if obj.disappeared_frames > self.max_disappeared:
                    del self.objects[obj.track_id]
            return list(self.objects.values())

        if not self.objects:
            for det in detections:
                tv = TrackedVehicle(self.next_track_id, det)
                self.objects[self.next_track_id] = tv
                self.next_track_id += 1
            return list(self.objects.values())

        # Match existing objects to current detections via Euclidean distance
        obj_ids = list(self.objects.keys())
        obj_centers = [self.objects[oid].positions[-1] for oid in obj_ids]
        det_centers = [d.center for d in detections]

        distances = np.linalg.norm(
            np.array(obj_centers)[:, np.newaxis] - np.array(det_centers), axis=2
        )

        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            if distances[r, c] > self.max_distance:
                continue

            oid = obj_ids[r]
            det = detections[c]
            self.objects[oid].bbox = det.bbox
            self.objects[oid].positions.append(det.center)
            self.objects[oid].disappeared_frames = 0
            used_rows.add(r)
            used_cols.add(c)

        # Mark disappeared
        for r, oid in enumerate(obj_ids):
            if r not in used_rows:
                self.objects[oid].disappeared_frames += 1
                if self.objects[oid].disappeared_frames > self.max_disappeared:
                    del self.objects[oid]

        # Register new objects
        for c, det in enumerate(detections):
            if c not in used_cols:
                tv = TrackedVehicle(self.next_track_id, det)
                self.objects[self.next_track_id] = tv
                self.next_track_id += 1

        return list(self.objects.values())


class VirtualTripwireCounter:
    """
    Virtual tripwire line counter.
    Counts vehicles crossing from upstream to downstream across a virtual line.
    """

    def __init__(self, line_y: int, direction_downwards: bool = True):
        self.line_y = line_y
        self.direction_downwards = direction_downwards
        self.counts = {"car": 0, "motorbike": 0, "bus": 0, "truck": 0}
        self.total_count = 0

    def update(self, tracked_vehicles: List[TrackedVehicle]) -> Dict[str, int]:
        for tv in tracked_vehicles:
            if tv.has_crossed or len(tv.positions) < 2:
                continue

            y_prev = tv.positions[-2][1]
            y_curr = tv.positions[-1][1]

            crossed = False
            if self.direction_downwards and y_prev <= self.line_y < y_curr:
                crossed = True
            elif not self.direction_downwards and y_prev >= self.line_y > y_curr:
                crossed = True

            if crossed:
                tv.has_crossed = True
                cls = tv.class_name if tv.class_name in self.counts else "car"
                self.counts[cls] += 1
                self.total_count += 1

        return dict(self.counts)

