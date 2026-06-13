from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from .models import (
    Location, Rider, Order, Alert,
    OrderStatus, RiderStatus, AlertSeverity,
)
from .geo import (
    generate_heatmap, GridCell, haversine_distance,
    distance_between, calculate_route_distance,
)
from .storage import store
from .config import settings


class OperationsService:
    def get_nearby_idle_riders(
        self,
        lat: float,
        lng: float,
        max_distance_meters: float = 3000.0,
        top_k: int = 10,
    ) -> List[dict]:
        idle_riders = store.list_riders(status=RiderStatus.IDLE)

        results = []
        for rider in idle_riders:
            if rider.current_location is None:
                continue

            dist = haversine_distance(
                lat, lng,
                rider.current_location.lat, rider.current_location.lng,
            )

            if dist <= max_distance_meters:
                results.append({
                    "rider": rider.model_dump(),
                    "distance_meters": round(dist, 1),
                })

        results.sort(key=lambda x: x["distance_meters"])
        return results[:top_k]

    def get_trajectory_replay(
        self,
        rider_id: str,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        sample_interval: int = 1,
    ) -> dict:
        rider = store.get_rider(rider_id)
        if not rider:
            return {"error": "rider_not_found"}

        trajectory = store.get_trajectory(rider_id, start_ts, end_ts)

        if not trajectory:
            return {
                "rider_id": rider_id,
                "rider_name": rider.name,
                "points": [],
                "total_distance_meters": 0,
                "duration_seconds": 0,
                "avg_speed_mps": 0,
            }

        if sample_interval > 1:
            trajectory = trajectory[::sample_interval]

        total_distance = calculate_route_distance(trajectory)
        duration = trajectory[-1].timestamp - trajectory[0].timestamp
        avg_speed = total_distance / duration if duration > 0 else 0

        orders = store.list_orders(rider_id=rider_id)
        order_timeline = []
        for order in orders:
            if order.assigned_at and (start_ts is None or order.assigned_at >= start_ts):
                if end_ts is None or order.assigned_at <= end_ts:
                    order_timeline.append({
                        "order_id": order.order_id,
                        "status": order.status.value,
                        "assigned_at": order.assigned_at,
                        "picked_at": order.picked_at,
                        "delivered_at": order.delivered_at,
                        "pickup_location": order.pickup_location.model_dump(),
                        "delivery_location": order.delivery_location.model_dump(),
                    })

        return {
            "rider_id": rider_id,
            "rider_name": rider.name,
            "points": [p.model_dump() for p in trajectory],
            "total_distance_meters": round(total_distance, 1),
            "duration_seconds": round(duration, 1),
            "avg_speed_mps": round(avg_speed, 2),
            "point_count": len(trajectory),
            "orders": order_timeline,
        }

    def get_heatmap_data(
        self,
        region: Optional[str] = None,
        since_ts: Optional[float] = None,
        grid_size: Optional[int] = None,
    ) -> dict:
        if grid_size is None:
            grid_size = settings.heatmap_grid_size

        locations = store.get_all_locations(since_ts=since_ts)

        if region:
            region_riders = {r.rider_id for r in store.list_riders(region=region)}
            rider_points = [p for p in store.trajectory_points if p.rider_id in region_riders]
            if since_ts:
                rider_points = [p for p in rider_points if p.timestamp >= since_ts]
            locations.extend(rider_points)

        if not locations:
            return {
                "grid": [],
                "total_points": 0,
                "max_count": 0,
                "grid_size": grid_size,
            }

        grid = generate_heatmap(locations, grid_size=grid_size)

        counts = [cell.count for cell in grid]
        max_count = max(counts) if counts else 0

        return {
            "grid": [
                {
                    "lat_center": cell.lat_center,
                    "lng_center": cell.lng_center,
                    "lat_min": cell.lat_min,
                    "lat_max": cell.lat_max,
                    "lng_min": cell.lng_min,
                    "lng_max": cell.lng_max,
                    "count": cell.count,
                    "intensity": cell.count / max_count if max_count > 0 else 0,
                }
                for cell in grid if cell.count > 0
            ],
            "total_points": len(locations),
            "max_count": max_count,
            "grid_size": grid_size,
        }

    def detect_abnormal_deliveries(
        self,
        lookback_minutes: int = 60,
    ) -> List[dict]:
        now = datetime.utcnow().timestamp()
        lookback_ts = now - lookback_minutes * 60

        recent_delivered = [
            o for o in store.list_orders(status=OrderStatus.DELIVERED)
            if o.delivered_at and o.delivered_at >= lookback_ts
        ]

        if not recent_delivered:
            return []

        delivery_times = []
        for o in recent_delivered:
            if o.assigned_at and o.delivered_at:
                dt = o.delivered_at - o.assigned_at
                delivery_times.append(dt)

        if not delivery_times:
            return []

        mean_dt = np.mean(delivery_times)
        std_dt = np.std(delivery_times)
        threshold = mean_dt + 2 * std_dt

        abnormal = []
        for order in recent_delivered:
            if order.assigned_at and order.delivered_at:
                dt = order.delivered_at - order.assigned_at
                if dt > threshold and std_dt > 60:
                    rider = store.get_rider(order.rider_id) if order.rider_id else None
                    abnormal.append({
                        "order": order.model_dump(),
                        "delivery_minutes": round(dt / 60, 1),
                        "expected_minutes": round(mean_dt / 60, 1),
                        "std_minutes": round(std_dt / 60, 1),
                        "z_score": round((dt - mean_dt) / std_dt, 2) if std_dt > 0 else 0,
                        "rider_name": rider.name if rider else None,
                    })

        abnormal.sort(key=lambda x: x["z_score"], reverse=True)
        return abnormal

    def detect_stationary_riders(
        self,
        stationary_minutes: int = 10,
        movement_threshold_meters: float = 50.0,
    ) -> List[dict]:
        now = datetime.utcnow().timestamp()
        lookback_ts = now - stationary_minutes * 60

        results = []
        for rider in store.list_riders():
            if rider.status == RiderStatus.OFFLINE:
                continue

            trajectory = store.get_trajectory(rider.rider_id, start_ts=lookback_ts)
            if len(trajectory) < 2:
                continue

            total_movement = calculate_route_distance(trajectory)
            first_point = trajectory[0]
            last_point = trajectory[-1]
            displacement = distance_between(first_point, last_point)

            if total_movement < movement_threshold_meters and rider.status == RiderStatus.IDLE:
                results.append({
                    "rider": rider.model_dump(),
                    "stationary_minutes": round((now - first_point.timestamp) / 60, 1),
                    "total_movement_meters": round(total_movement, 1),
                    "displacement_meters": round(displacement, 1),
                    "point_count": len(trajectory),
                })

        return results

    def get_system_alerts_summary(self) -> dict:
        all_alerts = store.list_alerts()
        unresolved = store.list_alerts(resolved=False)

        by_severity: Dict[str, int] = defaultdict(int)
        by_type: Dict[str, int] = defaultdict(int)

        for alert in unresolved:
            by_severity[alert.severity.value] += 1
            by_type[alert.type.value] += 1

        critical_count = sum(
            1 for a in unresolved if a.severity == AlertSeverity.CRITICAL
        )

        return {
            "total_alerts": len(all_alerts),
            "unresolved_alerts": len(unresolved),
            "critical_alerts": critical_count,
            "by_severity": dict(by_severity),
            "by_type": dict(by_type),
        }

    def get_dispatch_efficiency(
        self,
        lookback_minutes: int = 60,
    ) -> dict:
        now = datetime.utcnow().timestamp()
        lookback_ts = now - lookback_minutes * 60

        recent_orders = [
            o for o in store.orders.values()
            if o.created_at >= lookback_ts
        ]

        pending = [o for o in recent_orders if o.status == OrderStatus.PENDING]
        assigned = [o for o in recent_orders if o.status in [OrderStatus.ASSIGNED, OrderStatus.PICKED]]
        delivered = [o for o in recent_orders if o.status == OrderStatus.DELIVERED]

        avg_dispatch_time = 0.0
        dispatch_times = []
        for o in assigned + delivered:
            if o.assigned_at:
                dt = o.assigned_at - o.created_at
                dispatch_times.append(dt)
        if dispatch_times:
            avg_dispatch_time = np.mean(dispatch_times)

        avg_delivery_time = 0.0
        delivery_times = []
        for o in delivered:
            if o.assigned_at and o.delivered_at:
                dt = o.delivered_at - o.assigned_at
                delivery_times.append(dt)
        if delivery_times:
            avg_delivery_time = np.mean(delivery_times)

        return {
            "lookback_minutes": lookback_minutes,
            "total_orders": len(recent_orders),
            "pending": len(pending),
            "in_progress": len(assigned),
            "delivered": len(delivered),
            "avg_dispatch_seconds": round(avg_dispatch_time, 1),
            "avg_delivery_seconds": round(avg_delivery_time, 1),
            "auto_dispatch_rate": round(
                sum(1 for r in store.dispatch_records
                    if r.assigned_at >= lookback_ts and r.reason == "auto") / max(1, len(dispatch_times)),
                2
            ),
        }

    def bulk_reassign_orders(
        self,
        from_rider_id: str,
        to_rider_id: str,
        order_ids: Optional[List[str]] = None,
    ) -> dict:
        from_rider = store.get_rider(from_rider_id)
        to_rider = store.get_rider(to_rider_id)

        if not from_rider or not to_rider:
            return {"success": False, "error": "rider_not_found"}

        if to_rider.status == RiderStatus.OFFLINE:
            return {"success": False, "error": "target_rider_offline"}

        if order_ids is None:
            rider_orders = store.list_orders(
                rider_id=from_rider_id,
                status=OrderStatus.ASSIGNED,
            )
            order_ids = [o.order_id for o in rider_orders]

        reassigned = []
        failed = []
        available_capacity = to_rider.capacity - to_rider.current_orders

        for oid in order_ids:
            if available_capacity <= 0:
                failed.append({
                    "order_id": oid,
                    "reason": "target_rider_at_capacity",
                })
                continue

            order = store.get_order(oid)
            if not order:
                failed.append({"order_id": oid, "reason": "order_not_found"})
                continue

            if order.rider_id != from_rider_id:
                failed.append({"order_id": oid, "reason": "order_not_assigned_to_source"})
                continue

            if order.status not in [OrderStatus.ASSIGNED, OrderStatus.PICKED]:
                failed.append({"order_id": oid, "reason": "order_not_in_progress"})
                continue

            order.rider_id = to_rider_id
            from_rider.current_orders = max(0, from_rider.current_orders - 1)
            to_rider.current_orders += 1
            available_capacity -= 1

            reassigned.append({
                "order_id": oid,
                "from_rider": from_rider_id,
                "to_rider": to_rider_id,
            })

        return {
            "success": True,
            "reassigned": reassigned,
            "failed": failed,
            "total": len(order_ids),
        }

    def get_live_dashboard_data(self) -> dict:
        riders = store.list_riders()
        orders = store.list_orders()

        online_riders = [r for r in riders if r.status != RiderStatus.OFFLINE]
        idle_riders = [r for r in riders if r.status == RiderStatus.IDLE]
        busy_riders = [r for r in riders if r.status == RiderStatus.BUSY]

        pending_orders = [o for o in orders if o.status == OrderStatus.PENDING]
        in_progress = [o for o in orders if o.status in [OrderStatus.ASSIGNED, OrderStatus.PICKED]]
        delivered_today = [o for o in orders if o.status == OrderStatus.DELIVERED]

        overload_riders = [
            r for r in online_riders
            if r.current_orders >= r.capacity
        ]

        avg_current_load = np.mean([r.current_orders for r in online_riders]) if online_riders else 0

        return {
            "riders": {
                "total": len(riders),
                "online": len(online_riders),
                "idle": len(idle_riders),
                "busy": len(busy_riders),
                "overload": len(overload_riders),
                "avg_current_load": round(avg_current_load, 2),
            },
            "orders": {
                "total": len(orders),
                "pending": len(pending_orders),
                "in_progress": len(in_progress),
                "delivered_today": len(delivered_today),
            },
            "timestamp": datetime.utcnow().timestamp(),
        }


operations_service = OperationsService()
