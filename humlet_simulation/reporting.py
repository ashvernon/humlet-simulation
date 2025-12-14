from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt


class ReportGenerator:
    def __init__(self, db_path: Path, output_dir: Path) -> None:
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "charts").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT * FROM snapshots ORDER BY tick ASC")
        snap_rows = cur.fetchall()
        snap_cols = [d[0] for d in cur.description]

        cur.execute("SELECT cause, age, energy, health FROM deaths")
        deaths = cur.fetchall()

        cur.execute("SELECT * FROM run_meta")
        run_meta = cur.fetchone()
        meta_cols = [d[0] for d in cur.description]
        conn.close()

        snapshots = [dict(zip(snap_cols, row)) for row in snap_rows]
        meta = dict(zip(meta_cols, run_meta)) if run_meta else {}
        return snapshots, deaths, meta

    # ------------------------------------------------------------------ #
    def plot_population(self, snapshots):
        if not snapshots:
            return None
        ticks = [s["tick"] for s in snapshots]
        pop = [s["population"] for s in snapshots]
        births = [s.get("births", 0) for s in snapshots]
        deaths = [s.get("deaths", 0) for s in snapshots]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ticks, pop, label="Population", color="#4e79a7")
        ax.bar(ticks, births, label="Births", alpha=0.4, color="#59a14f")
        ax.bar(ticks, [-d for d in deaths], label="Deaths", alpha=0.4, color="#e15759")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Count")
        ax.legend()
        path = self.output_dir / "charts" / "population.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_death_causes(self, deaths):
        if not deaths:
            return None
        cause_counts = {}
        for cause, *_ in deaths:
            cause_counts[cause] = cause_counts.get(cause, 0) + 1

        labels = list(cause_counts)
        sizes = [cause_counts[c] for c in labels]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(sizes, labels=labels, autopct="%1.0f%%")
        ax.set_title("Deaths by cause")
        path = self.output_dir / "charts" / "death_causes.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_age_at_death(self, deaths):
        ages = [row[1] for row in deaths if row[1] is not None]
        if not ages:
            return None
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(ages, bins=15, color="#9c755f", edgecolor="black")
        ax.set_xlabel("Age at death")
        ax.set_ylabel("Count")
        path = self.output_dir / "charts" / "age_at_death.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    def write_html(self, meta, charts):
        html_path = self.output_dir / "summary.html"
        parts = ["<html><head><title>Humlet Telemetry Report</title></head><body>"]
        parts.append("<h1>Run summary</h1>")
        parts.append("<ul>")
        for key, value in meta.items():
            parts.append(f"<li><b>{key}</b>: {value}</li>")
        parts.append("</ul>")

        for title, path in charts:
            if path is None:
                continue
            rel = Path(path).name if Path(path).parent == self.output_dir else Path("charts") / Path(path).name
            parts.append(f"<h2>{title}</h2><img src='{rel}' alt='{title}' style='max-width: 100%;'>")

        parts.append("</body></html>")
        html_path.write_text("\n".join(parts), encoding="utf-8")
        return html_path

    def generate(self):
        snapshots, deaths, meta = self.load_data()
        charts = [
            ("Population over time", self.plot_population(snapshots)),
            ("Deaths by cause", self.plot_death_causes(deaths)),
            ("Age at death", self.plot_age_at_death(deaths)),
        ]
        return self.write_html(meta, charts)


def generate_report(db_path: Path, output_dir: Path) -> Path:
    generator = ReportGenerator(db_path, output_dir)
    return generator.generate()


__all__ = ["generate_report", "ReportGenerator"]
