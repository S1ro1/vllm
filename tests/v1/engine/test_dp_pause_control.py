# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from types import SimpleNamespace

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import DPEngineCoreProc, EngineCore


def test_dp_engine_add_request_notifies_coordinator_for_current_wave(monkeypatch):
    engine = object.__new__(DPEngineCoreProc)
    outputs = []
    engine.has_coordinator = True
    engine.current_wave = 7
    engine.engines_running = False
    engine.scheduler = SimpleNamespace(pause_state=PauseState.UNPAUSED)
    engine.output_queue = SimpleNamespace(put_nowait=outputs.append)
    monkeypatch.setattr(EngineCore, "add_request", lambda *args, **kwargs: None)

    DPEngineCoreProc.add_request(engine, SimpleNamespace(), request_wave=7)

    assert engine.engines_running
    assert outputs[0][0] == -1
    assert outputs[0][1].start_wave == 7


def test_dp_pause_ack_waits_for_engine_idle(monkeypatch):
    engine = object.__new__(DPEngineCoreProc)
    outputs = []
    pause_future: Future[None] = Future()
    engine.output_queue = SimpleNamespace(put_nowait=outputs.append)
    monkeypatch.setattr(
        engine, "pause_scheduler", lambda mode, clear_cache: pause_future
    )

    DPEngineCoreProc._handle_client_request(
        engine, EngineCoreRequestType.PAUSE_DP, (11, "keep", False)
    )
    assert outputs == []

    pause_future.set_result(None)
    assert outputs[0][0] == -1
    assert outputs[0][1].dp_pause_complete == 11


def test_dp_resume_ack_does_not_start_wave_before_coordinator(monkeypatch):
    engine = object.__new__(DPEngineCoreProc)
    outputs = []
    resume_calls = []
    engine.output_queue = SimpleNamespace(put_nowait=outputs.append)
    monkeypatch.setattr(
        engine,
        "_resume_scheduler",
        lambda start_dp_wave: resume_calls.append(start_dp_wave),
    )
    monkeypatch.setattr(engine, "has_work", lambda: True)

    DPEngineCoreProc._handle_client_request(engine, EngineCoreRequestType.RESUME_DP, 12)

    assert resume_calls == [False]
    assert outputs[0][0] == -1
    assert outputs[0][1].dp_resume_complete == (12, True)
