from __future__ import annotations

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import (
    Rider, Order, Location, Alert,
    RiderStatus, OrderStatus, AlertSeverity,
)
from app.storage import store
from app.location_service import location_service
from app.dispatch import dispatch_service
from app.operations import operations_service
from app.websocket_manager import ws_manager
from app.geo import get_region_for_location

app = FastAPI(title="外卖配送实时调度系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_snapshot_task: Optional[asyncio.Task] = None
_monitor_task: Optional[asyncio.Task] = None


def _ensure_snapshot_dir() -> None:
    Path(settings.snapshot_dir).mkdir(parents=True, exist_ok=True)


async def _snapshot_loop() -> None:
    while True:
        try:
            _ensure_snapshot_dir()
            now = datetime.utcnow()
            filename = f"snapshot_{now.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(settings.snapshot_dir, filename)
            data = store.to_dict()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            snapshots = sorted(Path(settings.snapshot_dir).glob("snapshot_*.json"))
            if len(snapshots) > 24:
                for old_snap in snapshots[:-24]:
                    old_snap.unlink()
        except Exception:
            pass
        await asyncio.sleep(settings.snapshot_interval_seconds)


async def _monitor_loop() -> None:
    while True:
        try:
            disconnect_alerts = location_service.check_disconnections()
            overload_alerts = location_service.check_overload()
            delay_alerts = location_service.check_delayed_orders()

            all_alerts = disconnect_alerts + overload_alerts + delay_alerts
            for alert in all_alerts:
                store.add_alert(alert)
                asyncio.create_task(ws_manager.broadcast_alert(alert))

            stats = operations_service.get_live_dashboard_data()
            asyncio.create_task(ws_manager.broadcast_stats_update(stats))

            dispatch_service.reset_dispatch_counters()
        except Exception:
            pass
        await asyncio.sleep(30)


@app.on_event("startup")
async def startup_event() -> None:
    global _snapshot_task, _monitor_task
    _ensure_snapshot_dir()
    _snapshot_task = asyncio.create_task(_snapshot_loop())
    _monitor_task = asyncio.create_task(_monitor_loop())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global _snapshot_task, _monitor_task
    if _snapshot_task:
        _snapshot_task.cancel()
    if _monitor_task:
        _monitor_task.cancel()


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "service": "外卖配送实时调度系统", "timestamp": datetime.utcnow().timestamp()}


@app.get("/health")
async def health_check() -> dict:
    return {"status": "healthy", "connections": ws_manager.get_connection_count()}


@app.post("/api/riders", response_model=Rider)
async def create_rider(rider: Rider) -> Rider:
    if rider.region == "default" and rider.current_location:
        rider.region = get_region_for_location(
            rider.current_location.lat, rider.current_location.lng
        )
    store.add_rider(rider)
    return rider


@app.get("/api/riders")
async def list_riders(
    status: Optional[RiderStatus] = None,
    region: Optional[str] = None,
) -> dict:
    riders = store.list_riders(status=status, region=region)
    return {
        "total": len(riders),
        "riders": [r.model_dump() for r in riders],
    }


@app.get("/api/riders/{rider_id}")
async def get_rider(rider_id: str) -> dict:
    rider = store.get_rider(rider_id)
    if not rider:
        raise HTTPException(status_code=404, detail="rider_not_found")
    eta = location_service.get_rider_eta(rider_id)
    return {
        **rider.model_dump(),
        "estimated_finish_seconds": eta,
    }


@app.put("/api/riders/{rider_id}/status")
async def update_rider_status(rider_id: str, status: RiderStatus) -> dict:
    rider = store.update_rider_status(rider_id, status)
    if not rider:
        raise HTTPException(status_code=404, detail="rider_not_found")
    if rider.current_location:
        asyncio.create_task(ws_manager.broadcast_rider_location(
            rider_id, rider.current_location, status, rider.current_orders
        ))
    return rider.model_dump()


@app.post("/api/riders/{rider_id}/reset-daily")
async def reset_rider_daily(rider_id: str) -> dict:
    rider = store.get_rider(rider_id)
    if not rider:
        raise HTTPException(status_code=404, detail="rider_not_found")
    rider.daily_orders = 0
    return {"success": True, "rider_id": rider_id, "daily_orders": 0}


@app.post("/api/riders/reset-daily-batch")
async def reset_all_riders_daily(region: Optional[str] = None) -> dict:
    riders = store.list_riders(region=region)
    count = 0
    for rider in riders:
        rider.daily_orders = 0
        count += 1
    return {"success": True, "reset_count": count}


@app.post("/api/riders/{rider_id}/location")
async def report_location(rider_id: str, location: Location) -> dict:
    smoothed, alerts = location_service.process_location_update(rider_id, location)
    if smoothed is None:
        return {"accepted": False, "alerts": [a.model_dump() for a in alerts]}

    rider = store.get_rider(rider_id)
    if rider:
        asyncio.create_task(ws_manager.broadcast_rider_location(
            rider_id, smoothed, rider.status, rider.current_orders
        ))

    for alert in alerts:
        store.add_alert(alert)
        asyncio.create_task(ws_manager.broadcast_alert(alert))

    return {
        "accepted": True,
        "location": smoothed.model_dump(),
        "alerts": [a.model_dump() for a in alerts],
    }


@app.get("/api/riders/{rider_id}/trajectory")
async def get_rider_trajectory(
    rider_id: str,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    sample_interval: int = 1,
) -> dict:
    result = operations_service.get_trajectory_replay(
        rider_id, start_ts, end_ts, sample_interval
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/api/orders", response_model=Order)
async def create_order(order: Order) -> Order:
    if order.region == "default":
        order.region = get_region_for_location(
            order.pickup_location.lat, order.pickup_location.lng
        )
    store.add_order(order)
    asyncio.create_task(ws_manager.broadcast_order_status(
        order.order_id, order.status, order.rider_id, order.estimated_delivery_seconds
    ))
    return order


@app.get("/api/orders")
async def list_orders(
    status: Optional[OrderStatus] = None,
    rider_id: Optional[str] = None,
    region: Optional[str] = None,
) -> dict:
    orders = store.list_orders(status=status, rider_id=rider_id, region=region)
    return {
        "total": len(orders),
        "orders": [o.model_dump() for o in orders],
    }


@app.get("/api/orders/{order_id}")
async def get_order(order_id: str) -> Order:
    order = store.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="order_not_found")
    return order


@app.put("/api/orders/{order_id}/status")
async def update_order_status(order_id: str, status: OrderStatus) -> dict:
    order = store.update_order_status(order_id, status)
    if not order:
        raise HTTPException(status_code=404, detail="order_not_found")
    asyncio.create_task(ws_manager.broadcast_order_status(
        order_id, status, order.rider_id, order.estimated_delivery_seconds
    ))
    return order.model_dump()


@app.post("/api/orders/{order_id}/dispatch")
async def dispatch_order(
    order_id: str,
    rider_id: Optional[str] = None,
) -> dict:
    order = store.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="order_not_found")
    if order.status != OrderStatus.PENDING:
        raise HTTPException(status_code=400, detail="order_not_pending")

    rider, record, info = dispatch_service.dispatch_order(order, manual_rider_id=rider_id)
    if not rider or not record:
        raise HTTPException(status_code=400, detail=info.get("error", "dispatch_failed"))

    return {
        "success": True,
        "rider": rider.model_dump(),
        "record": record.model_dump(),
        **info,
    }


@app.post("/api/orders/batch-dispatch")
async def batch_dispatch_orders(
    order_ids: List[str],
    strategy: str = "nearest",
) -> dict:
    if strategy not in ["nearest", "optimize"]:
        raise HTTPException(status_code=400, detail="invalid_strategy")
    return dispatch_service.batch_dispatch(order_ids, strategy=strategy)


@app.get("/api/orders/{order_id}/dispatch-candidates")
async def get_dispatch_candidates(
    order_id: str,
    top_k: int = 5,
) -> dict:
    candidates = dispatch_service.get_dispatch_candidates(order_id, top_k=top_k)
    return {"order_id": order_id, "candidates": candidates}


@app.get("/api/operations/nearby-riders")
async def get_nearby_idle_riders(
    lat: float,
    lng: float,
    max_distance_meters: float = 3000.0,
    top_k: int = 10,
) -> dict:
    results = operations_service.get_nearby_idle_riders(
        lat, lng, max_distance_meters=max_distance_meters, top_k=top_k
    )
    return {"total": len(results), "riders": results}


@app.get("/api/operations/heatmap")
async def get_heatmap(
    region: Optional[str] = None,
    since_ts: Optional[float] = None,
    grid_size: Optional[int] = None,
) -> dict:
    return operations_service.get_heatmap_data(
        region=region, since_ts=since_ts, grid_size=grid_size
    )


@app.get("/api/operations/abnormal-deliveries")
async def get_abnormal_deliveries(lookback_minutes: int = 60) -> dict:
    results = operations_service.detect_abnormal_deliveries(
        lookback_minutes=lookback_minutes
    )
    return {"total": len(results), "abnormal_deliveries": results}


@app.get("/api/operations/stationary-riders")
async def get_stationary_riders(
    stationary_minutes: int = 10,
    movement_threshold_meters: float = 50.0,
) -> dict:
    results = operations_service.detect_stationary_riders(
        stationary_minutes=stationary_minutes,
        movement_threshold_meters=movement_threshold_meters,
    )
    return {"total": len(results), "stationary_riders": results}


@app.get("/api/operations/dashboard")
async def get_live_dashboard() -> dict:
    return operations_service.get_live_dashboard_data()


@app.get("/api/operations/alerts")
async def list_alerts(
    resolved: Optional[bool] = None,
    severity: Optional[AlertSeverity] = None,
) -> dict:
    alerts = store.list_alerts(resolved=resolved, severity=severity)
    return {"total": len(alerts), "alerts": [a.model_dump() for a in alerts]}


@app.post("/api/operations/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str) -> dict:
    alert = store.resolve_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="alert_not_found")
    return alert.model_dump()


@app.get("/api/operations/alerts-summary")
async def alerts_summary() -> dict:
    return operations_service.get_system_alerts_summary()


@app.post("/api/operations/bulk-reassign")
async def bulk_reassign_orders(
    from_rider_id: str,
    to_rider_id: str,
    order_ids: Optional[List[str]] = None,
) -> dict:
    result = operations_service.bulk_reassign_orders(
        from_rider_id, to_rider_id, order_ids=order_ids
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "reassign_failed"))
    return result


@app.get("/api/operations/dispatch-efficiency")
async def get_dispatch_efficiency(lookback_minutes: int = 60) -> dict:
    return operations_service.get_dispatch_efficiency(lookback_minutes=lookback_minutes)


@app.get("/api/regions")
async def get_regions() -> dict:
    regions = store.get_regions()
    return {"regions": regions}


@app.get("/api/regions/load")
async def get_region_load() -> dict:
    return dispatch_service.get_region_load()


@app.get("/api/regions/stats")
async def get_region_stats() -> dict:
    regions = store.get_regions()
    stats_list = []

    for region in regions:
        riders = store.list_riders(region=region)
        online_riders = [r for r in riders if r.status != RiderStatus.OFFLINE]
        orders = store.list_orders(region=region)
        delivered = [o for o in orders if o.status == OrderStatus.DELIVERED]
        pending = [o for o in orders if o.status == OrderStatus.PENDING]
        overload = [r for r in online_riders if r.current_orders >= r.capacity]

        delivery_times = []
        for o in delivered:
            if o.assigned_at and o.delivered_at:
                delivery_times.append((o.delivered_at - o.assigned_at) / 60.0)

        avg_delivery = sum(delivery_times) / len(delivery_times) if delivery_times else 0.0

        active_rider_count = len(online_riders)
        avg_daily_orders = (
            sum(r.daily_orders for r in riders) / max(1, len(riders))
        )

        stats_list.append({
            "region": region,
            "rider_count": len(riders),
            "online_rider_count": len(online_riders),
            "total_orders": len(orders),
            "delivered_orders": len(delivered),
            "pending_orders": len(pending),
            "overload_count": len(overload),
            "avg_delivery_minutes": round(avg_delivery, 2),
            "avg_daily_orders_per_rider": round(avg_daily_orders, 2),
            "total_capacity": sum(r.capacity for r in online_riders),
            "current_load": sum(r.current_orders for r in online_riders),
        })

    return {
        "total_regions": len(stats_list),
        "stats": stats_list,
    }


@app.post("/api/snapshot/save")
async def save_snapshot() -> dict:
    _ensure_snapshot_dir()
    now = datetime.utcnow()
    filename = f"snapshot_{now.strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(settings.snapshot_dir, filename)
    data = store.to_dict()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"success": True, "filename": filename, "path": filepath}


@app.post("/api/snapshot/load")
async def load_snapshot(filename: str) -> dict:
    filepath = os.path.join(settings.snapshot_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="snapshot_not_found")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    store.load_from_dict(data)
    return {"success": True, "loaded": filename}


@app.get("/api/snapshots")
async def list_snapshots() -> dict:
    if not os.path.exists(settings.snapshot_dir):
        return {"snapshots": []}
    snapshots = sorted(
        [
            f.name for f in Path(settings.snapshot_dir).glob("snapshot_*.json")
        ],
        reverse=True,
    )
    return {"total": len(snapshots), "snapshots": snapshots}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str) -> None:
    await ws_manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()
            await ws_manager.handle_message(client_id, data)
            if data.get("action") == "subscribe_all":
                await ws_manager.send_initial_state(client_id)
    except WebSocketDisconnect:
        await ws_manager.disconnect(client_id)
    except Exception:
        await ws_manager.disconnect(client_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
