import json
from types import SimpleNamespace

import pytest

from src import week1_runner as runner


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    for folder in ("src", "prompts", "data/week1/envs"):
        (tmp_path / folder).mkdir(parents=True)
    contents = {"src/example.py": "VALUE = 1\n", "prompts/extract.txt": "strict original prompt\n",
        "uv.lock": "frozen dependencies", "pyproject.toml": "project",
        "data/week1/prepared.json": "{}", "data/week1/selection.json": '{"examples":[]}',
        "data/week1/envs/mem0-requirements.lock": "mem0 pinned",
        "data/week1/envs/memorybank-requirements.lock": "memorybank pinned"}
    for name, text in contents.items():
        (tmp_path / name).write_text(text)
    sources = {"mem0": {"commit": "pinned-mem0"}, "memorybank": {"commit": "pinned-memorybank"}}
    monkeypatch.setattr(runner, "baseline_sources", lambda root: json.loads(json.dumps(sources)))
    monkeypatch.setattr(runner, "snapshot_authors", lambda *args: None)
    monkeypatch.setattr(runner, "Gateway", lambda *args, **kwargs: pytest.fail("freeze/resume opened a paid gateway"))
    config = SimpleNamespace(model_dump=lambda: {"model": "fixed"})
    directory, manifest = runner.freeze_run(tmp_path, config)
    return tmp_path, directory, manifest, config, sources


def test_same_protocol_resumes_without_opening_model_clients(frozen):
    root, directory, manifest, config, _ = frozen
    runner.verify_resume(root, directory, manifest, config)


@pytest.mark.parametrize("name", ["src/example.py", "prompts/extract.txt", "uv.lock", "pyproject.toml"])
def test_source_prompt_and_dependency_changes_require_a_new_freeze(frozen, name):
    root, directory, manifest, config, _ = frozen
    with (root / name).open("a") as stream:
        stream.write("changed")
    with pytest.raises(ValueError, match="protocol changed"):
        runner.verify_resume(root, directory, manifest, config)


@pytest.mark.parametrize("operation", ["add", "delete"])
def test_source_file_set_is_part_of_the_frozen_protocol(frozen, operation):
    root, directory, manifest, config, _ = frozen
    if operation == "add":
        (root / "src/extra.py").write_text("NEW = True\n")
    else:
        (root / "src/example.py").unlink()
    with pytest.raises(ValueError, match="protocol changed"):
        runner.verify_resume(root, directory, manifest, config)


def test_saved_snapshot_tampering_is_rejected(frozen):
    root, directory, manifest, config, _ = frozen
    (directory / "source/example.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="snapshot changed"):
        runner.verify_resume(root, directory, manifest, config)


@pytest.mark.parametrize("saved", [False, True])
def test_live_and_frozen_question_lists_must_both_match(frozen, saved):
    root, directory, manifest, config, _ = frozen
    path = directory / "selection.json" if saved else root / "data/week1/selection.json"
    path.write_text("changed questions")
    with pytest.raises(ValueError, match="selection changed"):
        runner.verify_resume(root, directory, manifest, config)


def test_old_amendment_receipt_cannot_override_a_new_frozen_prompt(frozen):
    root, directory, manifest, config, _ = frozen
    (directory / "extraction-prompt-amendment.json").write_text(json.dumps({
        "scope": "ours-extraction-current-chunk-self-check", "authorized_by": "user"}))
    (root / "prompts/extract.txt").write_text("different prompt")
    with pytest.raises(ValueError, match="protocol changed"):
        runner.verify_resume(root, directory, manifest, config)


def test_old_task_protocol_requires_explicit_validated_baseline_import(frozen):
    root, directory, manifest, config, _ = frozen
    manifest["task_protocol"] = "old-global-canary-driver"
    with pytest.raises(ValueError, match="current frozen task protocol"):
        runner.verify_resume(root, directory, manifest, config)


def test_model_change_and_author_commit_change_cannot_resume(frozen):
    root, directory, manifest, config, sources = frozen
    different = SimpleNamespace(model_dump=lambda: {"model": "different"})
    with pytest.raises(ValueError, match="configuration changed"):
        runner.verify_resume(root, directory, manifest, different)
    sources["mem0"]["commit"] = "unapproved"
    with pytest.raises(ValueError, match="baseline sources"):
        runner.verify_resume(root, directory, manifest, config)


@pytest.fixture
def amendable(frozen):
    root, directory, manifest, config, sources = frozen
    (root / "src/week1_gateway.py").write_text("RETRIES = 0\n")
    directory, manifest = runner.freeze_run(root, config)
    return root, directory, manifest, config


def test_operational_code_amendment_resumes_with_the_same_protocol_fingerprint(amendable):
    root, directory, manifest, config = amendable
    (root / "src/week1_gateway.py").write_text("RETRIES = 6\n")
    with pytest.raises(ValueError, match="protocol changed"):
        runner.verify_resume(root, directory, manifest, config)
    result = runner.amend_code(root, directory, "add bounded retries")
    assert result == {"amended": True, "number": 1, "changed": ["src/week1_gateway.py"]}
    amended = json.loads((directory / "manifest.json").read_text())
    assert amended["protocol_sha256"] == manifest["protocol_sha256"]
    assert amended["frozen_files"] == manifest["frozen_files"]
    assert (directory / "amendments/1/src/week1_gateway.py").read_text() == "RETRIES = 6\n"
    runner.verify_resume(root, directory, amended, config)
    assert runner.amend_code(root, directory, "no change") == {"amended": False}


@pytest.mark.parametrize("name", ["src/example.py", "prompts/extract.txt", "uv.lock"])
def test_code_amendment_cannot_touch_protocol_files(amendable, name):
    root, directory, manifest, config = amendable
    with (root / name).open("a") as stream:
        stream.write("changed")
    with pytest.raises(ValueError, match="protocol files"):
        runner.amend_code(root, directory, "sneak a protocol change")
    assert "code_amendments" not in json.loads((directory / "manifest.json").read_text())


def test_tampered_amendment_snapshot_is_rejected(amendable):
    root, directory, manifest, config = amendable
    (root / "src/week1_gateway.py").write_text("RETRIES = 6\n")
    runner.amend_code(root, directory, "add bounded retries")
    (directory / "amendments/1/src/week1_gateway.py").write_text("RETRIES = 99\n")
    with pytest.raises(ValueError, match="amendment snapshot changed"):
        runner.verify_resume(root, directory, json.loads((directory / "manifest.json").read_text()), config)
