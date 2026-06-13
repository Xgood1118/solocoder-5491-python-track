from __future__ import annotations

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np

from .models import Location, LocationPoint


EARTH_RADIUS_METERS = 6371000.0


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_METERS * c


def distance_between(p1: Location | LocationPoint, p2: Location | LocationPoint) -> float:
    return haversine_distance(p1.lat, p1.lng, p2.lat, p2.lng)


def estimate_delivery_time_seconds(
    pickup: LocationPoint,
    dropoff: LocationPoint,
    rider_location: Optional[Location] = None,
    avg_speed_mps: float = 5.0,
    pickup_wait_seconds: int = 180,
    dropoff_wait_seconds: int = 120,
) -> int:
    total_distance = 0.0

    if rider_location is not None:
        total_distance += distance_between(rider_location, pickup)

    total_distance += distance_between(pickup, dropoff)

    travel_seconds = total_distance / avg_speed_mps if avg_speed_mps > 0 else 0
    total_seconds = travel_seconds + pickup_wait_seconds + dropoff_wait_seconds

    return int(max(60, total_seconds))


@dataclass
class GridCell:
    lat_min: float
    lat_max: float
    lng_min: float
    lng_max: float
    lat_center: float
    lng_center: float
    count: int = 0


def generate_heatmap(
    locations: List[Location],
    grid_size: int = 20,
    lat_range: Optional[Tuple[float, float]] = None,
    lng_range: Optional[Tuple[float, float]] = None,
) -> List[GridCell]:
    if not locations:
        return []

    lats = np.array([loc.lat for loc in locations])
    lngs = np.array([loc.lng for loc in locations])

    if lat_range is None:
        lat_range = (float(lats.min()), float(lats.max()))
    if lng_range is None:
        lng_range = (float(lngs.min()), float(lngs.max()))

    lat_min, lat_max = lat_range
    lng_min, lng_max = lng_range

    if lat_max - lat_min < 0.001:
        lat_max = lat_min + 0.01
    if lng_max - lng_min < 0.001:
        lng_max = lng_min + 0.01

    lat_step = (lat_max - lat_min) / grid_size
    lng_step = (lng_max - lng_min) / grid_size

    grid: List[GridCell] = []

    for i in range(grid_size):
        for j in range(grid_size):
            cell_lat_min = lat_min + i * lat_step
            cell_lat_max = cell_lat_min + lat_step
            cell_lng_min = lng_min + j * lng_step
            cell_lng_max = cell_lng_min + lng_step

            lat_mask = (lats >= cell_lat_min) & (lats < cell_lat_max)
            lng_mask = (lngs >= cell_lng_min) & (lngs < cell_lng_max)
            count = int(np.sum(lat_mask & lng_mask))

            grid.append(GridCell(
                lat_min=cell_lat_min,
                lat_max=cell_lat_max,
                lng_min=cell_lng_min,
                lng_max=cell_lng_max,
                lat_center=cell_lat_min + lat_step / 2,
                lng_center=cell_lng_min + lng_step / 2,
                count=count,
            ))

    return grid


def get_region_for_location(lat: float, lng: float, regions: Optional[dict] = None) -> str:
    if regions is None:
        regions = {
            "cbd": (39.90, 116.40, 0.05),
            "west": (39.91, 116.32, 0.08),
            "east": (39.92, 116.48, 0.08),
            "north": (39.98, 116.40, 0.08),
            "south": (39.85, 116.40, 0.08),
        }

    for region_name, (center_lat, center_lng, radius) in regions.items():
        dist = haversine_distance(lat, lng, center_lat, center_lng)
        if dist <= radius * 1000:
            return region_name

    return "default"


def detect_gps_drift(
    current: Location,
    previous: Optional[Location],
    threshold_meters: float = 50.0,
    max_speed_mps: float = 30.0,
) -> Tuple[bool, str]:
    if previous is None:
        return False, "no_previous"

    dist = distance_between(previous, current)
    time_delta = current.timestamp - previous.timestamp

    if time_delta <= 0:
        return True, "invalid_timestamp"

    speed = dist / time_delta

    if dist > threshold_meters * 5:
        return True, f"jump_too_large:{dist:.1f}m"

    if speed > max_speed_mps:
        return True, f"speed_too_high:{speed:.1f}m/s"

    if current.accuracy is not None and current.accuracy > 50:
        return True, f"low_accuracy:{current.accuracy:.1f}m"

    return False, "ok"


def smooth_location(
    current: Location,
    history: List[Location],
    window_size: int = 3,
) -> Location:
    if len(history) < 2:
        return current

    recent = history[-window_size:]
    weights = np.linspace(0.5, 1.0, len(recent))
    weights = weights / weights.sum()

    lats = np.array([loc.lat for loc in recent] + [current.lat])
    lngs = np.array([loc.lng for loc in recent] + [current.lng])

    w = np.append(weights, 0.5)
    w = w / w.sum()

    smoothed_lat = float(np.sum(lats * w))
    smoothed_lng = float(np.sum(lngs * w))

    return Location(
        lat=smoothed_lat,
        lng=smoothed_lng,
        timestamp=current.timestamp,
        accuracy=current.accuracy,
        speed=current.speed,
        bearing=current.bearing,
    )


def find_nearest_riders(
    target_lat: float,
    target_lng: float,
    riders: List,
    max_distance_meters: float = 5000.0,
    top_k: int = 10,
) -> List[Tuple]:
    results = []

    for rider in riders:
        if rider.current_location is None:
            continue

        dist = haversine_distance(
            target_lat, target_lng,
            rider.current_location.lat, rider.current_location.lng
        )

        if dist <= max_distance_meters:
            results.append((rider, dist))

    results.sort(key=lambda x: x[1])
    return results[:top_k]


def calculate_route_distance(points: List[Location | LocationPoint]) -> float:
    if len(points) < 2:
        return 0.0

    total = 0.0
    for i in range(1, len(points)):
        total += distance_between(points[i - 1], points[i])

    return total
