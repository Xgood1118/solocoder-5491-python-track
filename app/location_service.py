from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple, List
from collections import defaultdict

from .models import Location, Rider, RiderStatus, OrderStatus, Alert, AlertType, AlertSeverity
from .geo import detect_gps_drift, smooth_location, distance_between
from .storage import store
from .config import settings


class LocationService:
    def __init__(self):
        self.last_locations: dict[str, Location] = {}
        self.disconnect_timers: dict[str, float] = {}
        self.consecutive_drifts: dict[str, int] = defaultdict(int)

    def process_location_update(
        self,
        rider_id: str,
        raw_location: Location,
    ) -> Tuple[Optional[Location], List[Alert]]:
        alerts: List[Alert] = []

        rider = store.get_rider(rider_id)
        if not rider:
            return None, alerts

        previous = self.last_locations.get(rider_id)

        is_drift, reason = detect_gps_drift(
            raw_location, previous,
            threshold_meters=settings.gps_drift_threshold_meters,
        )

        if is_drift:
            is_true_drift = reason.startswith("jump_too_large") or reason.startswith("speed_too_high")

            if is_true_drift:
                self.consecutive_drifts[rider_id] += 1
            else:
                self.consecutive_drifts[rider_id] = max(0, self.consecutive_drifts[rider_id] - 1)

            if self.consecutive_drifts[rider_id] >= 3:
                alerts.append(Alert(
                    type=AlertType.DRIFT,
                    severity=AlertSeverity.WARNING,
                    rider_id=rider_id,
                    message=f"骑手{rider_id} GPS连续漂移: {reason}",
                ))
                self.consecutive_drifts[rider_id] = 0
                return None, alerts

            if self.consecutive_drifts[rider_id] >= 2 and is_true_drift:
                history = list(store.location_history.get(rider_id, []))
                smoothed = smooth_location(raw_location, history, window_size=5)
                self._accept_location(rider_id, smoothed, rider, alerts)
                return smoothed, alerts

            return None, alerts

        self.consecutive_drifts[rider_id] = 0

        history = list(store.location_history.get(rider_id, []))
        smoothed = smooth_location(raw_location, history, window_size=3)

        self._accept_location(rider_id, smoothed, rider, alerts)

        return smoothed, alerts

    def _accept_location(
        self,
        rider_id: str,
        location: Location,
        rider: Rider,
        alerts: List[Alert],
    ) -> None:
        self.last_locations[rider_id] = location
        self.disconnect_timers[rider_id] = location.timestamp
        store.update_rider_location(rider_id, location)

        if rider.status == RiderStatus.OFFLINE:
            store.update_rider_status(rider_id, RiderStatus.IDLE)

        alerts.extend(self._check_speed_anomaly(rider, location))

    def _check_speed_anomaly(self, rider: Rider, location: Location) -> List[Alert]:
        alerts: List[Alert] = []

        if location.speed and location.speed > 25.0:
            alerts.append(Alert(
                type=AlertType.ABNORMAL,
                severity=AlertSeverity.INFO,
                rider_id=rider.rider_id,
                message=f"骑手{rider.name}速度异常: {location.speed:.1f}m/s",
            ))

        return alerts

    def check_disconnections(self) -> List[Alert]:
        alerts: List[Alert] = []
        now = datetime.utcnow().timestamp()

        for rider in store.list_riders():
            if rider.status == RiderStatus.OFFLINE:
                continue

            last_seen = self.disconnect_timers.get(rider.rider_id, rider.last_seen)
            idle_time = now - last_seen

            if idle_time > settings.disconnect_timeout_seconds:
                if rider.status != RiderStatus.OFFLINE:
                    store.update_rider_status(rider.rider_id, RiderStatus.OFFLINE)
                    alerts.append(Alert(
                        type=AlertType.DISCONNECT,
                        severity=AlertSeverity.WARNING,
                        rider_id=rider.rider_id,
                        message=f"骑手{rider.name}已断线 {int(idle_time)}秒",
                    ))

        return alerts

    def check_overload(self) -> List[Alert]:
        alerts: List[Alert] = []

        for rider in store.list_riders():
            if rider.status == RiderStatus.OFFLINE:
                continue

            load_ratio = rider.current_orders / rider.capacity if rider.capacity > 0 else 0

            if load_ratio >= 0.9:
                unresolved = store.list_alerts(
                    resolved=False,
                    alert_type=AlertType.OVERLOAD,
                )
                already_alerted = any(a.rider_id == rider.rider_id for a in unresolved)

                if not already_alerted:
                    severity = AlertSeverity.CRITICAL if load_ratio >= 1.0 else AlertSeverity.WARNING
                    alerts.append(Alert(
                        type=AlertType.OVERLOAD,
                        severity=severity,
                        rider_id=rider.rider_id,
                        message=(
                            f"骑手{rider.name}超负荷: "
                            f"{rider.current_orders}/{rider.capacity}单"
                        ),
                    ))

        return alerts

    def check_delayed_orders(self) -> List[Alert]:
        alerts: List[Alert] = []
        now = datetime.utcnow().timestamp()

        for order in store.list_orders():
            if order.status not in [OrderStatus.ASSIGNED, OrderStatus.PICKED]:
                continue

            if order.assigned_at and order.estimated_delivery_seconds:
                elapsed = now - order.assigned_at
                if elapsed > order.estimated_delivery_seconds * 1.2:
                    unresolved = store.list_alerts(
                        resolved=False,
                        alert_type=AlertType.DELAY,
                    )
                    already_alerted = any(a.order_id == order.order_id for a in unresolved)

                    if not already_alerted:
                        alerts.append(Alert(
                            type=AlertType.DELAY,
                            severity=AlertSeverity.WARNING,
                            order_id=order.order_id,
                            rider_id=order.rider_id,
                            message=f"订单{order.order_id}可能超时: 已{int(elapsed)}秒",
                        ))

        return alerts

    def get_rider_eta(self, rider_id: str) -> Optional[int]:
        rider = store.get_rider(rider_id)
        if not rider or rider.current_location is None:
            return None

        orders = store.list_orders(status=OrderStatus.ASSIGNED, rider_id=rider_id)
        if not orders:
            return 0

        total_eta = 0
        for order in orders:
            if order.estimated_delivery_seconds:
                if order.status == OrderStatus.PICKED:
                    elapsed = datetime.utcnow().timestamp() - (order.picked_at if order.picked_at else order.assigned_at)
                    remaining = max(0, order.estimated_delivery_seconds - elapsed)
                    total_eta += remaining
                else:
                    total_eta += order.estimated_delivery_seconds

        return int(total_eta)

    def cleanup(self) -> None:
        pass


location_service = LocationService()
