from pathlib import Path
import sys

import pytest

pytest.importorskip("numpy")
pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from humlet_simulation.reporting import generate_report
from humlet_simulation.telemetry import DeathEvent, TelemetryRecorder


def test_generate_report_creates_files(tmp_path):
    recorder = TelemetryRecorder(
        "report_run",
        base_seed=99,
        world_size=(10, 10),
        snapshot_interval=1,
        base_path=tmp_path,
    )

    class DummyEnv:
        def __init__(self):
            self.objects = []
            self.regions = []
            self.cols = 1
            self.rows = 1

    env = DummyEnv()

    recorder.record_snapshot(
        tick=0,
        humlets=[],
        env=env,
        births=0,
        deaths=0,
    )

    recorder.log_deaths(
        [
            DeathEvent(
                tick=0,
                humlet_id=1,
                family_id=1,
                generation=0,
                age=5,
                x=0.0,
                y=0.0,
                region_col=None,
                region_row=None,
                energy=0.0,
                health=0.0,
                hunger_need=0.0,
                safety_need=0.0,
                social_need=0.0,
                cause="starvation",
                last_action=None,
                brain_outputs=None,
            )
        ]
    )

    db_path = recorder.db_path
    output_dir = recorder.run_dir

    report_path = generate_report(db_path, output_dir)

    assert report_path.exists()
    charts_dir = output_dir / "charts"
    assert charts_dir.exists()
    # At least one chart should exist when we have snapshot + death rows
    assert any(charts_dir.iterdir())
