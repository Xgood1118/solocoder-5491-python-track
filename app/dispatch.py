from __future__ import annotations

import asyncio
import numpy as np
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from collections import defaultdict

from .models import (
    Order, Rider, DispatchRecord,
    RiderStatus, OrderStatus,
)
from .geo import (
    haversine_distance, distance_between,
    estimate_delivery_time_seconds, find_nearest_riders,
)
from .storage import store
from .config import settings
from .websocket_manager import ws_manager


class DispatchService:
    def __init__(self):
        self.dispatch_history: Dict[str, int] = defaultdict(int)
        self.region_load_balance: Dict[str, Dict[str, float]] = defaultdict(dict)

    def calculate_rider_score(
        self,
        rider: Rider,
        order: Order,
    ) -> Tuple[float, dict]:
        if rider.current_location is None:
            return -1.0, {"reason": "no_location"}

        if rider.status == RiderStatus.OFFLINE:
            return -1.0, {"reason": "offline"}

        if rider.current_orders >= rider.capacity:
            return -1.0, {"reason": "over_capacity"}

        if order.region != rider.region and rider.region != "default":
            pass

        pickup_dist = distance_between(rider.current_location, order.pickup_location)
        delivery_dist = distance_between(order.pickup_location, order.delivery_location)
        total_dist = pickup_dist + delivery_dist

        max_dist = 10000.0
        dist_score = max(0.0, 1.0 - (total_dist / max_dist))

        rating_score = rider.score / 5.0

        load_ratio = rider.current_orders / rider.capacity
        load_score = 1.0 - load_ratio

        recent_dispatches = self.dispatch_history.get(rider.rider_id, 0)
        recent_penalty = min(0.3, recent_dispatches * 0.05)

        avg_speed = 1.0
        if rider.avg_delivery_minutes > 0:
            avg_speed = 30.0 / max(5.0, rider.avg_delivery_minutes)

        weights = {
            "distance": 0.45,
            "rating": 0.25,
            "load": 0.20,
            "speed": 0.10,
        }

        final_score = (
            dist_score * weights["distance"]
            + rating_score * weights["rating"]
            + load_score * weights["load"]
            + min(1.0, avg_speed) * weights["speed"]
            - recent_penalty
        )

        breakdown = {
            "distance_score": round(dist_score, 3),
            "rating_score": round(rating_score, 3),
            "load_score": round(load_score, 3),
            "speed_score": round(min(1.0, avg_speed), 3),
            "recent_penalty": round(recent_penalty, 3),
            "total_distance_meters": round(total_dist, 1),
            "pickup_distance_meters": round(pickup_dist, 1),
            "final_score": round(final_score, 3),
        }

        return max(0.0, final_score), breakdown

    def find_best_rider(
        self,
        order: Order,
        max_candidates: int = 10,
    ) -> Tuple[Optional[Rider], Optional[DispatchRecord], dict]:
        available_riders = store.list_riders()
        available_riders = [
            r for r in available_riders
            if r.status != RiderStatus.OFFLINE
            and r.current_location is not None
            and r.current_orders < r.capacity
        ]

        if not available_riders:
            return None, None, {"error": "no_available_riders"}

        nearby = find_nearest_riders(
            order.pickup_location.lat,
            order.pickup_location.lng,
            available_riders,
            max_distance_meters=8000.0,
            top_k=max_candidates,
        )

        if not nearby:
            cbd_riders = [r for r in available_riders if r.region == "cbd"]
            non_cbd = [r for r in available_riders if r.region != "cbd"]
            if len(cbd_riders) < len(available_riders) * 0.3:
                candidates = available_riders[:max_candidates]
            else:
                candidates = non_cbd[:max_candidates] if non_cbd else cbd_riders[:max_candidates]
        else:
            candidates = [r for r, d in nearby]

        scores: List[Tuple[Rider, float, dict]] = []
        for rider in candidates:
            score, breakdown = self.calculate_rider_score(rider, order)
            if score > 0:
                scores.append((rider, score, breakdown))

        if not scores:
            return None, None, {"error": "no_qualified_riders"}

        scores.sort(key=lambda x: x[1], reverse=True)
        best_rider, best_score, breakdown = scores[0]

        pickup_dist = distance_between(best_rider.current_location, order.pickup_location)

        record = DispatchRecord(
            order_id=order.order_id,
            rider_id=best_rider.rider_id,
            distance_meters=round(pickup_dist, 1),
            score=round(best_score, 3),
            reason=breakdown.get("reason", "auto"),
        )

        return best_rider, record, breakdown

    def dispatch_order(
        self,
        order: Order,
        manual_rider_id: Optional[str] = None,
    ) -> Tuple[Optional[Rider], Optional[DispatchRecord], dict]:
        if manual_rider_id:
            rider = store.get_rider(manual_rider_id)
            if not rider:
                return None, None, {"error": "rider_not_found"}
            if rider.status == RiderStatus.OFFLINE:
                return None, None, {"error": "rider_offline"}
            if rider.current_orders >= rider.capacity:
                return None, None, {"error": "rider_over_capacity"}

            if rider.current_location:
                dist = distance_between(rider.current_location, order.pickup_location)
            else:
                dist = 0.0

            record = DispatchRecord(
                order_id=order.order_id,
                rider_id=rider.rider_id,
                distance_meters=round(dist, 1),
                score=1.0,
                reason="manual",
            )
        else:
            rider, record, breakdown = self.find_best_rider(order)
            if not rider or not record:
                return None, None, breakdown

        eta = estimate_delivery_time_seconds(
            order.pickup_location,
            order.delivery_location,
            rider_location=rider.current_location,
        )

        store.update_order_status(
            order.order_id,
            OrderStatus.ASSIGNED,
            rider_id=rider.rider_id,
            estimated_delivery_seconds=eta,
        )

        store.add_dispatch_record(record)
        self.dispatch_history[rider.rider_id] += 1

        asyncio.create_task(ws_manager.broadcast_order_status(
            order.order_id,
            OrderStatus.ASSIGNED,
            rider_id=rider.rider_id,
            estimated_delivery_seconds=eta,
        ))

        asyncio.create_task(ws_manager.broadcast_dispatch_update({
            "order_id": order.order_id,
            "rider_id": rider.rider_id,
            "rider_name": rider.name,
            "distance_meters": record.distance_meters,
            "eta_seconds": eta,
            "score": record.score,
        }))

        return rider, record, {"eta_seconds": eta}

    def batch_dispatch(
        self,
        order_ids: List[str],
        strategy: str = "nearest",
    ) -> dict:
        results = {
            "success": [],
            "failed": [],
            "total": len(order_ids),
        }

        orders = []
        for oid in order_ids:
            order = store.get_order(oid)
            if order and order.status == OrderStatus.PENDING:
                orders.append(order)
            else:
                results["failed"].append({
                    "order_id": oid,
                    "reason": "invalid_or_not_pending",
                })

        if strategy == "nearest":
            for order in orders:
                rider, record, info = self.dispatch_order(order)
                if rider and record:
                    results["success"].append({
                        "order_id": order.order_id,
                        "rider_id": rider.rider_id,
                        "rider_name": rider.name,
                        **info,
                    })
                else:
                    results["failed"].append({
                        "order_id": order.order_id,
                        **info,
                    })

        elif strategy == "optimize":
            results = self._optimized_batch_dispatch(orders, results)

        return results

    def _optimized_batch_dispatch(
        self,
        orders: List[Order],
        results: dict,
    ) -> dict:
        if not orders:
            return results

        rider_order_matrix: Dict[str, List[Tuple[Order, float, dict]]] = defaultdict(list)

        for order in orders:
            available = store.list_riders()
            available = [
                r for r in available
                if r.status != RiderStatus.OFFLINE
                and r.current_location is not None
                and r.current_orders + len(rider_order_matrix[r.rider_id]) < r.capacity
            ]

            for rider in available:
                score, breakdown = self.calculate_rider_score(rider, order)
                if score > 0:
                    rider_order_matrix[rider.rider_id].append((order, score, breakdown))

        assigned: set = set()

        for rider_id, assignments in rider_order_matrix.items():
            assignments.sort(key=lambda x: x[1], reverse=True)
            rider = store.get_rider(rider_id)
            if not rider:
                continue

            max_assign = rider.capacity - rider.current_orders
            for order, score, breakdown in assignments[:max_assign]:
                if order.order_id in assigned:
                    continue

                order = store.get_order(order.order_id)
                if not order or order.status != OrderStatus.PENDING:
                    continue

                eta = estimate_delivery_time_seconds(
                    order.pickup_location,
                    order.delivery_location,
                    rider_location=rider.current_location,
                )

                dist = distance_between(rider.current_location, order.pickup_location)
                record = DispatchRecord(
                    order_id=order.order_id,
                    rider_id=rider.rider_id,
                    distance_meters=round(dist, 1),
                    score=round(score, 3),
                    reason="batch_optimize",
                )

                store.update_order_status(
                    order.order_id,
                    OrderStatus.ASSIGNED,
                    rider_id=rider.rider_id,
                    estimated_delivery_seconds=eta,
                )
                store.add_dispatch_record(record)

                assigned.add(order.order_id)
                results["success"].append({
                    "order_id": order.order_id,
                    "rider_id": rider.rider_id,
                    "rider_name": rider.name,
                    "eta_seconds": eta,
                    "distance_meters": round(dist, 1),
                })

                asyncio.create_task(ws_manager.broadcast_order_status(
                    order.order_id,
                    OrderStatus.ASSIGNED,
                    rider_id=rider.rider_id,
                    estimated_delivery_seconds=eta,
                ))

        for order in orders:
            if order.order_id not in assigned:
                results["failed"].append({
                    "order_id": order.order_id,
                    "reason": "no_available_rider",
                })

        return results

    def get_dispatch_candidates(
        self,
        order_id: str,
        top_k: int = 5,
    ) -> List[dict]:
        order = store.get_order(order_id)
        if not order:
            return []

        available_riders = store.list_riders()
        available_riders = [
            r for r in available_riders
            if r.status != RiderStatus.OFFLINE
            and r.current_location is not None
            and r.current_orders < r.capacity
        ]

        results = []
        for rider in available_riders:
            score, breakdown = self.calculate_rider_score(rider, order)
            if score > 0:
                dist = distance_between(rider.current_location, order.pickup_location)
                eta = estimate_delivery_time_seconds(
                    order.pickup_location,
                    order.delivery_location,
                    rider_location=rider.current_location,
                )
                results.append({
                    "rider": rider.model_dump(),
                    "score": round(score, 3),
                    "pickup_distance_meters": round(dist, 1),
                    "eta_seconds": eta,
                    "breakdown": breakdown,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_region_load(self) -> Dict[str, dict]:
        regions = store.get_regions()
        load_data: Dict[str, dict] = {}

        for region in regions:
            riders = store.list_riders(region=region)
            online = [r for r in riders if r.status != RiderStatus.OFFLINE]
            total_capacity = sum(r.capacity for r in online)
            current_load = sum(r.current_orders for r in online)
            pending_orders = len(store.list_orders(status=OrderStatus.PENDING, region=region))

            load_ratio = current_load / total_capacity if total_capacity > 0 else 0

            load_data[region] = {
                "rider_count": len(riders),
                "online_rider_count": len(online),
                "total_capacity": total_capacity,
                "current_load": current_load,
                "load_ratio": round(load_ratio, 2),
                "pending_orders": pending_orders,
                "available_capacity": total_capacity - current_load,
            }

        return load_data

    def reset_dispatch_counters(self) -> None:
        self.dispatch_history.clear()


dispatch_service = DispatchService()
