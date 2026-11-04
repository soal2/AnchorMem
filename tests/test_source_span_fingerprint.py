from pathlib import Path

import pytest

from src import engine
from src.config import ServiceConfig, load_config


def test_source_mapping_changes_reject_existing_index_without_consolidation(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    config, path = load_config(root / "configs/service/v1-b3.yaml", ServiceConfig)
    config = config.model_copy(update={"database_dir": str(tmp_path / "index")})
    assert config.consolidation.mode == "off" and config.directory_selector is None
    first = engine.MemoryEngine(config, path, None, None)
    with first.store.connect("user"):
        pass
    original = Path.read_bytes
    source_rules = Path(engine.__file__).with_name("source_spans.py")

    def changed_source_rules(file):
        data = original(file)
        return data + b"\n# changed source mapping rules\n" if file == source_rules else data

    monkeypatch.setattr(Path, "read_bytes", changed_source_rules)
    with pytest.raises(ValueError, match="configuration|fingerprint"):
        with engine.MemoryEngine(config, path, None, None).store.connect("user"):
            pass
