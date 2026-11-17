"""CPU regressions for selecting a previously prepared checkpoint."""

import json

import pytest

from vllm_jev import clef_export, native_models


@pytest.mark.parametrize("model_id", sorted(clef_export.MODELS))
def test_clef_cache_belongs_to_requested_model(tmp_path, monkeypatch, model_id):
    output = tmp_path / "checkpoint" / model_id
    output.mkdir(parents=True)
    manifest = output / "clef_manifest.json"
    verified = []
    monkeypatch.setattr(
        clef_export,
        "verify_clef",
        lambda path, *, full: verified.append((path, full)),
    )
    monkeypatch.setattr(
        native_models,
        "snapshot_download",
        lambda **kwargs: pytest.fail("an existing checkpoint must not download"),
    )

    other_model = next(name for name in clef_export.MODELS if name != model_id)
    manifest.write_text(json.dumps({"source_repository": other_model}))
    with pytest.raises(ValueError, match="another repository"):
        native_models._prepare(model_id, tmp_path, "auto")
    assert not verified

    manifest.write_text(json.dumps({"source_repository": model_id}))
    assert native_models._prepare(model_id, tmp_path, "auto") == output
    assert verified == [(output, True)]


PRIVATE_CHECKPOINT = "example/private-checkpoint"
REVISION = "a" * 40


def test_cached_checkpoint_must_match_requested_revision(tmp_path, monkeypatch):
    output = tmp_path / "checkpoint" / PRIVATE_CHECKPOINT
    output.mkdir(parents=True)
    (output / "decision_manifest.json").write_text(
        json.dumps(
            {
                "source_repository": PRIVATE_CHECKPOINT,
                "source_revision": "b" * 40,
            }
        )
    )
    monkeypatch.setattr(native_models, "verify", lambda path, *, full: None)

    with pytest.raises(ValueError, match="is pinned"):
        native_models._prepare(PRIVATE_CHECKPOINT, tmp_path, "auto", REVISION)
    assert (
        native_models._prepare(PRIVATE_CHECKPOINT, tmp_path, "auto", "b" * 40) == output
    )


@pytest.mark.parametrize("requested", [False, True])
def test_hub_lookup_uses_requested_revision_and_token(tmp_path, monkeypatch, requested):
    model_id, pinned = next(iter(native_models.KEV_PINNED_REVISIONS.items()))
    options = (REVISION, True) if requested else ()
    looked_up = {}

    class Stop(Exception):
        pass

    class Api:
        def model_info(self, model_id, **kwargs):
            looked_up.update(kwargs)
            raise Stop

    monkeypatch.setattr(native_models, "HfApi", Api)
    with pytest.raises(Stop):
        native_models._prepare(model_id, tmp_path, "auto", *options)
    # Unrequested, the table's pin is looked up anonymously, as before.
    assert (looked_up["revision"], looked_up["token"]) == (options or (pinned, False))


def test_serve_passes_revision_and_required_token_to_prepare(tmp_path, monkeypatch):
    from vllm_jev import cli

    class Stop(Exception):
        pass

    forwarded = {}
    prepared = {}

    def run(command, **kwargs):
        forwarded["command"] = command
        raise Stop

    def prepare(model_id, workspace, protocol, **kwargs):
        prepared.update(kwargs)
        return tmp_path

    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.delenv("VLLM_JEV_BACKEND", raising=False)
    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "vllm-jev",
            "serve",
            PRIVATE_CHECKPOINT,
            "--workspace",
            str(tmp_path),
            "--revision",
            REVISION,
            "--hf-auth",
        ],
    )
    with pytest.raises(Stop):
        cli.main()
    command = forwarded["command"]
    assert command[1:3] == ["-m", "vllm_jev.native_models"]

    monkeypatch.setattr(native_models, "prepare", prepare)
    monkeypatch.setattr(native_models.sys, "argv", command[2:])
    native_models.main()
    assert prepared == {"revision": REVISION, "token": True}
