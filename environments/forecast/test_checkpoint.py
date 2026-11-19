# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT

from opentslm.forecast_checkpoint import publish, recover


def test_recover_ignores_incomplete_and_damaged_generations(tmp_path):
    identity = {"seed": 27}
    for step in (2, 4):
        staging = tmp_path / ".staging"
        staging.mkdir()
        (staging / "state").write_bytes(b"shard")
        publish(tmp_path, staging, step, identity)
    (tmp_path / ".partial").mkdir()
    assert recover(tmp_path, identity)[0] == 4
    (tmp_path / "step-00000004" / "state").write_bytes(b"wrong")
    assert recover(tmp_path, identity)[0] == 2
    assert recover(tmp_path, {"seed": 13}) is None
