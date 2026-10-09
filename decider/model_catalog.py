"""Pinned, data-only local models eligible for a serving-session switch."""
import os
from pathlib import Path

from decider.startlux import MODEL, REVISION

PRESETS = (
    dict(id=MODEL, name="StartLux Decision 0.8B", revision=REVISION, license="CC-BY-NC-4.0"),
    dict(id="Mapika/decider-0.8b", name="Mapika Decider 0.8B",
         revision="a0a01d6f8135298f400a8c856b355793012ae971"),
)


def validate_directory(path):
    path = Path(path)
    configs = [name for name in ("decider_config.json", "decision_config.json") if (path / name).is_file()]
    if not path.is_dir() or not (path / "config.json").is_file() or len(configs) != 1:
        raise ValueError("Incomplete or ambiguous model folder; config.json and exactly one decision config are required")
    if not any(path.glob("*.safetensors")):
        raise ValueError("No safetensors weights in model folder")
    return str(path.resolve())


def cached_path(preset):
    from huggingface_hub import snapshot_download
    return validate_directory(snapshot_download(
        repo_id=preset["id"], revision=preset["revision"],
        cache_dir=os.environ.get("DECIDER_MODEL_CACHE") or None,
        local_files_only=True))


class Catalog:
    def __init__(self, startup, name=None):
        self.startup = startup
        self.paths = {}
        self.startup_name = name or "启动配置模型"
        self.rows = []
        self.active = "startup"
        self.refresh()

    def refresh(self):
        rows, paths = [], {"startup": self.startup}
        for preset in PRESETS:
            row = dict(preset, available=False)
            try:
                path = cached_path(preset)
            except (OSError, ValueError):
                pass
            else:
                row["available"] = True
                paths[preset["id"]] = path
                if self.startup == preset["id"] or Path(self.startup).resolve() == Path(path):
                    paths["startup"] = path
                    if self.active == "startup":
                        self.active = preset["id"]
            rows.append(row)
        if paths["startup"] not in [paths.get(row["id"]) for row in rows]:
            rows.insert(0, dict(id="startup", name="启动配置 · " + self.startup_name, available=True))
        self.paths, self.rows = paths, rows
        return rows

    def resolve(self, model_id):
        self.refresh()
        if model_id not in self.paths:
            raise ValueError("Model is unknown or not installed in the serving cache; install the pinned checkpoint first")
        return self.paths[model_id]
