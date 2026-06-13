from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Deque
from datetime import datetime

from .models import (
    Rider, Order, Location, TrajectoryPoint, Alert,
    DispatchRecord, RiderStatus, OrderStatus,
    AlertType, AlertSeverity,
)
from .config import settings


class DataStore:
    _instance: Optional["DataStore"] = None
    _lock: threading.RLock

    riders: Dict[str, Rider]
    orders: Dict[str, Order]
    alerts: Dict[str, Alert]
    dispatch_records: List[DispatchRecord]

    location_history: Dict[str, Deque[Location]]
    trajectory_points: List[TrajectoryPoint]

    rider_location_subscribers: Dict[str, List]
    order_status_subscribers: Dict[str, List]
    global_subscribers: List

    def __new__(cls) -> "DataStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self._lock = threading.RLock()
        self.riders = {}
        self.orders = {}
        self.alerts = {}
        self.dispatch_records = []
        self.location_history = defaultdict(lambda: deque(maxlen=1000))
        self.trajectory_points = []
        self.rider_location_subscribers = defaultdict(list)
        self.order_status_subscribers = defaultdict(list)
        self.global_subscribers = []

    def add_rider(self, rider: Rider) -> Rider:
        with self._lock:
            self.riders[rider.rider_id] = rider
            return rider

    def get_rider(self, rider_id: str) -> Optional[Rider]:
        with self._lock:
            return self.riders.get(rider_id)

    def list_riders(self, status: Optional[RiderStatus] = None, region: Optional[str] = None) -> List[Rider]:
        with self._lock:
            riders = list(self.riders.values())
            if status:
                riders = [r for r in riders if r.status == status]
            if region:
                riders = [r for r in riders if r.region == region]
            return riders

    def update_rider_location(self, rider_id: str, location: Location) -> Optional[Rider]:
        with self._lock:
            rider = self.riders.get(rider_id)
            if not rider:
                return None

            rider.current_location = location
            rider.last_seen = location.timestamp

            self.location_history[rider_id].append(location)
            self.trajectory_points.append(
                TrajectoryPoint(**location.model_dump(), rider_id=rider_id)
            )

            if len(self.trajectory_points) > 100000:
                self.trajectory_points = self.trajectory_points[-50000:]

            return rider

    def update_rider_status(self, rider_id: str, status: RiderStatus) -> Optional[Rider]:
        with self._lock:
            rider = self.riders.get(rider_id)
            if not rider:
                return None

            old_status = rider.status
            rider.status = status

            if status in [RiderStatus.IDLE, RiderStatus.BUSY] and old_status == RiderStatus.OFFLINE:
                rider.online_since = datetime.utcnow().timestamp()
            elif status == RiderStatus.OFFLINE:
                rider.online_since = None

            return rider

    def update_rider_load(self, rider_id: str, current_orders: int) -> Optional[Rider]:
        with self._lock:
            rider = self.riders.get(rider_id)
            if not rider:
                return None

            rider.current_orders = current_orders
            if current_orders >= rider.capacity:
                rider.status = RiderStatus.BUSY
            elif rider.status == RiderStatus.BUSY and current_orders < max(1, rider.capacity // 2):
                rider.status = RiderStatus.IDLE

            return rider

    def add_order(self, order: Order) -> Order:
        with self._lock:
            self.orders[order.order_id] = order
            return order

    def get_order(self, order_id: str) -> Optional[Order]:
        with self._lock:
            return self.orders.get(order_id)

    def list_orders(
        self,
        status: Optional[OrderStatus] = None,
        rider_id: Optional[str] = None,
        region: Optional[str] = None,
    ) -> List[Order]:
        with self._lock:
            orders = list(self.orders.values())
            if status:
                orders = [o for o in orders if o.status == status]
            if rider_id:
                orders = [o for o in orders if o.rider_id == rider_id]
            if region:
                orders = [o for o in orders if o.region == region]
            return orders

    def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        rider_id: Optional[str] = None,
        estimated_delivery_seconds: Optional[int] = None,
    ) -> Optional[Order]:
        with self._lock:
            order = self.orders.get(order_id)
            if not order:
                return None

            now = datetime.utcnow().timestamp()
            order.status = status

            if status == OrderStatus.ASSIGNED and rider_id:
                order.rider_id = rider_id
                order.assigned_at = now
                order.estimated_delivery_seconds = estimated_delivery_seconds
                rider = self.riders.get(rider_id)
                if rider:
                    rider.current_orders += 1
                    if rider.current_orders >= rider.capacity:
                        rider.status = RiderStatus.BUSY

            elif status == OrderStatus.PICKED:
                order.picked_at = now

            elif status == OrderStatus.DELIVERED:
                order.delivered_at = now
                if order.rider_id:
                    rider = self.riders.get(order.rider_id)
                    if rider:
                        rider.current_orders = max(0, rider.current_orders - 1)
                        rider.total_delivered += 1
                        rider.daily_orders += 1
                        if order.assigned_at and order.delivered_at:
                            delivery_minutes = (order.delivered_at - order.assigned_at) / 60.0
                            if rider.avg_delivery_minutes == 0:
                                rider.avg_delivery_minutes = delivery_minutes
                            else:
                                rider.avg_delivery_minutes = (
                                    rider.avg_delivery_minutes * 0.9 + delivery_minutes * 0.1
                                )
                        if rider.current_orders < max(1, rider.capacity // 2):
                            rider.status = RiderStatus.IDLE

            return order

    def add_alert(self, alert: Alert) -> Alert:
        with self._lock:
            self.alerts[alert.alert_id] = alert
            return alert

    def list_alerts(
        self,
        resolved: Optional[bool] = None,
        severity: Optional[AlertSeverity] = None,
        alert_type: Optional[AlertType] = None,
    ) -> List[Alert]:
        with self._lock:
            alerts = list(self.alerts.values())
            if resolved is not None:
                alerts = [a for a in alerts if a.resolved == resolved]
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            if alert_type:
                alerts = [a for a in alerts if a.type == alert_type]
            alerts.sort(key=lambda a: a.created_at, reverse=True)
            return alerts

    def resolve_alert(self, alert_id: str) -> Optional[Alert]:
        with self._lock:
            alert = self.alerts.get(alert_id)
            if alert:
                alert.resolved = True
            return alert

    def add_dispatch_record(self, record: DispatchRecord) -> DispatchRecord:
        with self._lock:
            self.dispatch_records.append(record)
            return record

    def get_trajectory(
        self,
        rider_id: str,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> List[Location]:
        with self._lock:
            history = list(self.location_history.get(rider_id, []))
            if start_ts:
                history = [h for h in history if h.timestamp >= start_ts]
            if end_ts:
                history = [h for h in history if h.timestamp <= end_ts]
            return history

    def get_all_locations(self, since_ts: Optional[float] = None) -> List[Location]:
        with self._lock:
            locations: List[Location] = []
            for rider in self.riders.values():
                if rider.current_location:
                    if since_ts is None or rider.current_location.timestamp >= since_ts:
                        locations.append(rider.current_location)
            return locations

    def get_regions(self) -> List[str]:
        with self._lock:
            regions = set()
            for rider in self.riders.values():
                regions.add(rider.region)
            for order in self.orders.values():
                regions.add(order.region)
            return sorted(regions)

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "riders": [r.model_dump() for r in self.riders.values()],
                "orders": [o.model_dump() for o in self.orders.values()],
                "alerts": [a.model_dump() for a in self.alerts.values()],
                "dispatch_records": [d.model_dump() for d in self.dispatch_records],
                "trajectory_points": [t.model_dump() for t in self.trajectory_points[-10000:]],
                "timestamp": datetime.utcnow().timestamp(),
            }

    def load_from_dict(self, data: dict) -> None:
        with self._lock:
            self._initialize()

            for rider_data in data.get("riders", []):
                rider = Rider(**rider_data)
                self.riders[rider.rider_id] = rider

            for order_data in data.get("orders", []):
                order = Order(**order_data)
                self.orders[order.order_id] = order

            for alert_data in data.get("alerts", []):
                alert = Alert(**alert_data)
                self.alerts[alert.alert_id] = alert

            for record_data in data.get("dispatch_records", []):
                self.dispatch_records.append(DispatchRecord(**record_data))

            for point_data in data.get("trajectory_points", []):
                point = TrajectoryPoint(**point_data)
                self.trajectory_points.append(point)
                self.location_history[point.rider_id].append(
                    Location(**{k: v for k, v in point_data.items() if k != "rider_id"})
                )


store = DataStore()
