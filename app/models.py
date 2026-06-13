from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field, field_validator


def _uid() -> str:
    return uuid.uuid4().hex[:16]


def _now_ts() -> float:
    return datetime.utcnow().timestamp()


class RiderStatus(str, Enum):
    OFFLINE = "offline"
    IDLE = "idle"
    BUSY = "busy"
    RESTING = "resting"


class OrderStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    PICKED = "picked"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    OVERLOAD = "overload"
    DISCONNECT = "disconnect"
    DRIFT = "drift"
    DELAY = "delay"
    ABNORMAL = "abnormal"


class Location(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="纬度")
    lng: float = Field(..., ge=-180, le=180, description="经度")
    timestamp: float = Field(default_factory=_now_ts, description="上报时间戳(秒)")
    accuracy: Optional[float] = Field(None, description="精度(米)")
    speed: Optional[float] = Field(None, description="速度(m/s)")
    bearing: Optional[float] = Field(None, description="方向角(度)")


class LocationPoint(BaseModel):
    lat: float
    lng: float


class Rider(BaseModel):
    rider_id: str = Field(default_factory=_uid)
    name: str
    phone: str
    region: str = Field(default="default", description="所属区域")
    status: RiderStatus = RiderStatus.OFFLINE
    current_location: Optional[Location] = None
    last_seen: float = Field(default_factory=_now_ts, description="最后活跃时间戳")
    score: float = Field(5.0, ge=0.0, le=5.0, description="历史评分")
    daily_orders: int = 0
    current_orders: int = 0
    total_delivered: int = 0
    avg_delivery_minutes: float = 0.0
    capacity: int = Field(8, ge=1, le=20, description="最大承载单数")
    online_since: Optional[float] = None

    @field_validator("score")
    @classmethod
    def _round_score(cls, v: float) -> float:
        return round(v, 2)


class Order(BaseModel):
    order_id: str = Field(default_factory=_uid)
    customer_name: str
    customer_phone: str
    pickup_address: str
    pickup_location: LocationPoint
    delivery_address: str
    delivery_location: LocationPoint
    region: str = "default"
    priority: int = Field(1, ge=1, le=5, description="优先级 1普通 5紧急")
    promised_delivery_seconds: int = 1800
    created_at: float = Field(default_factory=_now_ts)
    assigned_at: Optional[float] = None
    picked_at: Optional[float] = None
    delivered_at: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    rider_id: Optional[str] = None
    estimated_delivery_seconds: Optional[int] = None
    items: List[str] = Field(default_factory=list)


class TrajectoryPoint(Location):
    rider_id: str


class DispatchRecord(BaseModel):
    record_id: str = Field(default_factory=_uid)
    order_id: str
    rider_id: str
    assigned_at: float = Field(default_factory=_now_ts)
    distance_meters: float
    score: float
    reason: str = "auto"


class Alert(BaseModel):
    alert_id: str = Field(default_factory=_uid)
    type: AlertType
    severity: AlertSeverity
    rider_id: Optional[str] = None
    order_id: Optional[str] = None
    message: str
    created_at: float = Field(default_factory=_now_ts)
    resolved: bool = False


class RegionStats(BaseModel):
    region: str
    rider_count: int
    online_rider_count: int
    total_orders: int
    delivered_orders: int
    avg_delivery_minutes: float
    avg_daily_orders_per_rider: float
    overload_count: int
    pending_orders: int


class RiderLocationPush(BaseModel):
    type: str = "rider_location"
    rider_id: str
    location: Location
    status: RiderStatus
    current_orders: int


class OrderStatusPush(BaseModel):
    type: str = "order_status"
    order_id: str
    status: OrderStatus
    rider_id: Optional[str]
    estimated_delivery_seconds: Optional[int]


class AlertPush(BaseModel):
    type: str = "alert"
    alert: Alert


class DispatchPush(BaseModel):
    type: str = "dispatch_update"
    order_id: str
    rider_id: str
    rider_name: str
    distance_meters: float
    eta_seconds: Optional[int] = None
    score: float


class StatsPush(BaseModel):
    type: str = "stats_update"
    timestamp: float
    riders: dict
    orders: dict
