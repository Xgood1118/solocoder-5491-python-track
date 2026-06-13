from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    port: int = 8000
    host: str = "0.0.0.0"
    snapshot_dir: str = "./snapshots"
    snapshot_interval_seconds: int = 300
    gps_drift_threshold_meters: float = 50.0
    disconnect_timeout_seconds: int = 120
    max_orders_per_rider: int = 8
    heatmap_grid_size: int = 20

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
