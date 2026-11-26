"""Measure the energy and CO2 footprint of a run with CodeCarbon."""

import json
import logging
from pathlib import Path

from codecarbon import EmissionsTracker


def start_tracker(project_name: str) -> EmissionsTracker | None:
    """Start a tracker. Return None if CodeCarbon fails: never break a training."""
    try:
        tracker = EmissionsTracker(
            project_name=project_name,
            save_to_file=False,  # we write our own file next to the weights
            log_level="warning",
        )
        tracker.start()
        return tracker
    except Exception:
        logging.exception("CodeCarbon could not start, training is not measured")
        return None


def stop_and_save(tracker: EmissionsTracker | None, save_dir: Path) -> None:
    """Stop the tracker and write emissions.json in save_dir."""
    if tracker is None:
        return
    try:
        tracker.stop()
        data = dict(tracker.final_emissions_data.values)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "emissions.json").write_text(
            json.dumps(data, indent=2, default=str)
        )
        logging.info(
            f"Energy: {data['energy_consumed']:.4f} kWh, "
            f"CO2: {data['emissions']:.4f} kg"
        )
    except Exception:
        logging.exception("CodeCarbon could not save the emissions")