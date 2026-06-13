from __future__ import annotations

import json
import asyncio
from typing import Dict, List, Set, Optional, Any
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from .models import (
    RiderLocationPush, OrderStatusPush, AlertPush,
    RiderStatus, OrderStatus,
)
from .storage import store


class Connection:
    def __init__(self, websocket: WebSocket, client_id: str):
        self.websocket = websocket
        self.client_id = client_id
        self.subscribed_riders: Set[str] = set()
        self.subscribed_orders: Set[str] = set()
        self.subscribe_all: bool = False
        self.created_at: float = datetime.utcnow().timestamp()


class WebSocketManager:
    def __init__(self):
        self.connections: Dict[str, Connection] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str) -> Connection:
        await websocket.accept()
        conn = Connection(websocket, client_id)
        async with self._lock:
            self.connections[client_id] = conn
        return conn

    async def disconnect(self, client_id: str) -> None:
        async with self._lock:
            if client_id in self.connections:
                del self.connections[client_id]

    async def handle_message(self, client_id: str, message: dict) -> None:
        async with self._lock:
            conn = self.connections.get(client_id)
            if not conn:
                return

        action = message.get("action")
        if action == "subscribe_rider":
            rider_id = message.get("rider_id")
            if rider_id:
                async with self._lock:
                    conn.subscribed_riders.add(rider_id)
                await self._send_to_client(client_id, {
                    "type": "subscribed",
                    "entity": "rider",
                    "id": rider_id,
                })

        elif action == "unsubscribe_rider":
            rider_id = message.get("rider_id")
            if rider_id:
                async with self._lock:
                    conn.subscribed_riders.discard(rider_id)

        elif action == "subscribe_order":
            order_id = message.get("order_id")
            if order_id:
                async with self._lock:
                    conn.subscribed_orders.add(order_id)
                await self._send_to_client(client_id, {
                    "type": "subscribed",
                    "entity": "order",
                    "id": order_id,
                })

        elif action == "unsubscribe_order":
            order_id = message.get("order_id")
            if order_id:
                async with self._lock:
                    conn.subscribed_orders.discard(order_id)

        elif action == "subscribe_all":
            async with self._lock:
                conn.subscribe_all = True
            await self._send_to_client(client_id, {
                "type": "subscribed",
                "entity": "all",
            })

        elif action == "unsubscribe_all":
            async with self._lock:
                conn.subscribe_all = False
                conn.subscribed_riders.clear()
                conn.subscribed_orders.clear()

    async def broadcast_rider_location(
        self,
        rider_id: str,
        location,
        status: RiderStatus,
        current_orders: int,
    ) -> None:
        push = RiderLocationPush(
            rider_id=rider_id,
            location=location,
            status=status,
            current_orders=current_orders,
        )

        message = push.model_dump()

        async with self._lock:
            connections = list(self.connections.values())

        for conn in connections:
            should_send = conn.subscribe_all or rider_id in conn.subscribed_riders
            if should_send:
                await self._send_to_client(conn.client_id, message)

    async def broadcast_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        rider_id: Optional[str] = None,
        estimated_delivery_seconds: Optional[int] = None,
    ) -> None:
        push = OrderStatusPush(
            order_id=order_id,
            status=status,
            rider_id=rider_id,
            estimated_delivery_seconds=estimated_delivery_seconds,
        )

        message = push.model_dump()

        async with self._lock:
            connections = list(self.connections.values())

        for conn in connections:
            should_send = (
                conn.subscribe_all
                or order_id in conn.subscribed_orders
                or (rider_id and rider_id in conn.subscribed_riders)
            )
            if should_send:
                await self._send_to_client(conn.client_id, message)

    async def broadcast_alert(self, alert) -> None:
        push = AlertPush(alert=alert)
        message = push.model_dump()

        async with self._lock:
            connections = list(self.connections.values())

        for conn in connections:
            if conn.subscribe_all:
                await self._send_to_client(conn.client_id, message)

    async def broadcast_dispatch_update(self, data: dict) -> None:
        message = {
            "type": "dispatch_update",
            **data,
        }

        async with self._lock:
            connections = list(self.connections.values())

        for conn in connections:
            if conn.subscribe_all:
                await self._send_to_client(conn.client_id, message)

    async def _send_to_client(self, client_id: str, message: dict) -> None:
        async with self._lock:
            conn = self.connections.get(client_id)
            if not conn:
                return

        try:
            await conn.websocket.send_json(message)
        except (WebSocketDisconnect, RuntimeError):
            await self.disconnect(client_id)
        except Exception:
            pass

    async def send_initial_state(self, client_id: str) -> None:
        async with self._lock:
            conn = self.connections.get(client_id)
            if not conn:
                return

        if conn.subscribe_all:
            riders = store.list_riders()
            for rider in riders:
                if rider.current_location:
                    push = RiderLocationPush(
                        rider_id=rider.rider_id,
                        location=rider.current_location,
                        status=rider.status,
                        current_orders=rider.current_orders,
                    )
                    await self._send_to_client(client_id, push.model_dump())

            pending_orders = store.list_orders(status=OrderStatus.PENDING)
            for order in pending_orders[:20]:
                push = OrderStatusPush(
                    order_id=order.order_id,
                    status=order.status,
                    rider_id=order.rider_id,
                    estimated_delivery_seconds=order.estimated_delivery_seconds,
                )
                await self._send_to_client(client_id, push.model_dump())

    async def broadcast_stats_update(self, stats: dict) -> None:
        message = {
            "type": "stats_update",
            "timestamp": datetime.utcnow().timestamp(),
            **stats,
        }

        async with self._lock:
            connections = list(self.connections.values())

        for conn in connections:
            if conn.subscribe_all:
                await self._send_to_client(conn.client_id, message)

    def get_connection_count(self) -> int:
        return len(self.connections)


ws_manager = WebSocketManager()
