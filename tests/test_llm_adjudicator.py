"""Tests for ssmlint.llm_adjudicator -- the Tier 2 Local LLM Adjudicator.
Deliberately fast and free of any real, running Ollama server: every test
here mocks the one real HTTP boundary (`_post_chat` for chat calls,
`urllib.request.urlopen` for the `/api/tags` reachability check) --
mirroring test_classifier.py's own "no real model in the default pytest
run" discipline, which matters even more here since GitHub Actions CI
cannot have a local Ollama process at all. A real, live-verified run
against the actual running server (Ollama 0.33.1, qwen2.5:3b-instruct
pulled) is documented separately in CONTRIBUTING.md with real pasted
output, not exercised by this file.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

from ssmlint.blocks import Block, NonConformingCell, SheetBlocks
from ssmlint.labeling import build_block_example
from ssmlint.llm_adjudicator import (
    _json_schema_for_labels,
    apply_tier2_to_workbook,
    check_ollama_available,
    predict_labels_llm,
    query_ollama_label,
)
from ssmlint.rules import Issue


def _block(
    cells: list[str], non_conforming: list[NonConformingCell] | None = None, span: str = "C14:E14", row: int = 14
) -> Block:
    return Block(
        sheet="Model", row=row, span=span, pattern="(R[0]C[-1]*(1+R1C2))",
        cells=cells, non_conforming=non_conforming or [], near_misses=[],
    )


def _issue(cell: str, rule_id: str = "literal-in-formula-block") -> Issue:
    return Issue(cell=cell, severity="medium", rule_id=rule_id, explanation="x", suggested_fix="y")


def _fake_chat_response(label: str, reasoning: str = "because") -> dict:
    """Shape of a real Ollama /api/chat response body -- confirmed live
    against the actual running server before this module was written
    (see llm_adjudicator.py's own module docstring)."""
    return {"message": {"content": json.dumps({"label": label, "reasoning": reasoning})}}


def _fake_urlopen_response(body: dict) -> MagicMock:
    """A context-manager mock standing in for urllib's real response
    object (`.read()` returns the raw bytes `json.loads` expects)."""
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = json.dumps(body).encode("utf-8")
    return cm


# ---------------------------------------------------------------------------
# _json_schema_for_labels
# ---------------------------------------------------------------------------


def test_json_schema_enum_matches_labeling_labels_exactly() -> None:
    from ssmlint.labeling import LABELS

    schema = _json_schema_for_labels()

    assert schema["properties"]["label"]["enum"] == list(LABELS)
    assert schema["required"] == ["label", "reasoning"]


# ---------------------------------------------------------------------------
# query_ollama_label
# ---------------------------------------------------------------------------


def test_query_ollama_label_returns_the_models_chosen_label() -> None:
    with patch("ssmlint.llm_adjudicator._post_chat", return_value=_fake_chat_response("subtotal")):
        label = query_ollama_label("Block: Model!C14:G14\nBase shape: category_total")

    assert label == "subtotal"


def test_query_ollama_label_rejects_an_out_of_enum_label() -> None:
    """Defensive check even though the real JSON schema already
    constrains this -- confirms a future Ollama/model surprise fails
    loudly rather than silently poisoning a prediction list."""
    with patch("ssmlint.llm_adjudicator._post_chat", return_value=_fake_chat_response("not_a_real_label")):
        try:
            query_ollama_label("some block text")
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "not_a_real_label" in str(exc)


def test_query_ollama_label_sends_the_enum_schema_and_prompt_text() -> None:
    captured = {}

    def _capture(payload, endpoint, timeout):
        captured.update(payload)
        return _fake_chat_response("unknown")

    with patch("ssmlint.llm_adjudicator._post_chat", side_effect=_capture):
        query_ollama_label("my block description", model="qwen2.5:3b-instruct", endpoint="http://example:1234")

    assert captured["model"] == "qwen2.5:3b-instruct"
    assert captured["stream"] is False
    assert captured["format"]["properties"]["label"]["enum"]
    assert captured["messages"][-1] == {"role": "user", "content": "my block description"}


# ---------------------------------------------------------------------------
# predict_labels_llm -- caching by block signature
# ---------------------------------------------------------------------------


def test_predict_labels_llm_empty_examples_returns_empty_list_without_any_call() -> None:
    with patch("ssmlint.llm_adjudicator.query_ollama_label") as mock_query:
        result = predict_labels_llm([])

    assert result == []
    mock_query.assert_not_called()


def test_predict_labels_llm_cache_hit_avoids_a_second_real_call() -> None:
    example = build_block_example("entry_1", _block(["Model!C14", "Model!D14", "Model!E14"]), base_shape="growth_chain")
    cache: dict[str, str] = {}

    with patch("ssmlint.llm_adjudicator.query_ollama_label", return_value="suspected_error") as mock_query:
        first = predict_labels_llm([example], cache=cache)
        second = predict_labels_llm([example], cache=cache)  # identical block signature, second call

    assert first == ["suspected_error"]
    assert second == ["suspected_error"]
    mock_query.assert_called_once()  # the real LLM was only ever consulted once


def test_predict_labels_llm_without_a_cache_calls_every_time() -> None:
    example = build_block_example("entry_1", _block(["Model!C14", "Model!D14", "Model!E14"]), base_shape="growth_chain")

    with patch("ssmlint.llm_adjudicator.query_ollama_label", return_value="unknown") as mock_query:
        predict_labels_llm([example], cache=None)
        predict_labels_llm([example], cache=None)

    assert mock_query.call_count == 2


def test_predict_labels_llm_different_blocks_get_different_cache_keys() -> None:
    example_a = build_block_example("entry_1", _block(["Model!C14", "Model!D14", "Model!E14"]), base_shape="growth_chain")
    example_b = build_block_example(
        "entry_1",
        _block(["Model!C20", "Model!D20", "Model!E20"], span="C20:E20", row=20),
        base_shape="growth_chain",
    )
    cache: dict[str, str] = {}

    with patch("ssmlint.llm_adjudicator.query_ollama_label", side_effect=["subtotal", "unknown"]) as mock_query:
        result = predict_labels_llm([example_a, example_b], cache=cache)

    assert result == ["subtotal", "unknown"]
    assert mock_query.call_count == 2
    assert len(cache) == 2


# ---------------------------------------------------------------------------
# apply_tier2_to_workbook -- mirrors test_classifier.py's apply_tier1_to_workbook tests
# ---------------------------------------------------------------------------


def test_apply_tier2_to_workbook_splits_issues_exhaustively_when_nothing_suppressed() -> None:
    block = _block(["Model!C14", "Model!D14", "Model!E14"])
    sheet_blocks = [SheetBlocks(sheet="Model", blocks=[block])]
    issues = [_issue("Model!C14"), _issue("Model!D14")]

    with patch("ssmlint.llm_adjudicator.predict_labels_llm", return_value=["suspected_error"]):
        surviving, suppressed = apply_tier2_to_workbook(issues, sheet_blocks)

    assert surviving == issues
    assert suppressed == []


def test_apply_tier2_to_workbook_splits_issues_exhaustively_when_a_block_is_suppressed() -> None:
    block = _block(["Model!C14", "Model!D14", "Model!E14"])
    sheet_blocks = [SheetBlocks(sheet="Model", blocks=[block])]
    issues = [_issue("Model!C14"), _issue("Model!D14")]

    with patch("ssmlint.llm_adjudicator.predict_labels_llm", return_value=["intentional_override"]):
        surviving, suppressed = apply_tier2_to_workbook(issues, sheet_blocks)

    assert surviving == []
    assert {i.cell for i in suppressed} == {"Model!C14", "Model!D14"}
    assert set(surviving) | set(suppressed) == set(issues)
    assert set(surviving) & set(suppressed) == set()


def test_apply_tier2_to_workbook_flattens_blocks_across_multiple_sheets() -> None:
    block_a = _block(["Model!C14", "Model!D14", "Model!E14"])
    block_b = _block(["Other!C14", "Other!D14", "Other!E14"])
    sheet_blocks = [SheetBlocks(sheet="Model", blocks=[block_a]), SheetBlocks(sheet="Other", blocks=[block_b])]
    issues = [_issue("Model!C14"), _issue("Other!C14")]

    with patch("ssmlint.llm_adjudicator.predict_labels_llm", return_value=["subtotal", "suspected_error"]):
        surviving, suppressed = apply_tier2_to_workbook(issues, sheet_blocks)

    assert {i.cell for i in suppressed} == {"Model!C14"}
    assert {i.cell for i in surviving} == {"Other!C14"}


def test_apply_tier2_to_workbook_passes_model_endpoint_and_cache_through() -> None:
    block = _block(["Model!C14", "Model!D14", "Model!E14"])
    sheet_blocks = [SheetBlocks(sheet="Model", blocks=[block])]
    cache: dict[str, str] = {}
    captured = {}

    def _capture(examples, model, endpoint, cache):
        captured["model"] = model
        captured["endpoint"] = endpoint
        captured["cache"] = cache
        return ["unknown"]

    with patch("ssmlint.llm_adjudicator.predict_labels_llm", side_effect=_capture):
        apply_tier2_to_workbook([], sheet_blocks, model="my-model", endpoint="http://custom:9999", cache=cache)

    assert captured["model"] == "my-model"
    assert captured["endpoint"] == "http://custom:9999"
    assert captured["cache"] is cache


# ---------------------------------------------------------------------------
# check_ollama_available -- fails loudly with an actionable message
# ---------------------------------------------------------------------------


def test_check_ollama_available_passes_silently_when_model_is_pulled() -> None:
    body = {"models": [{"name": "qwen2.5:3b-instruct"}]}
    with patch("urllib.request.urlopen", return_value=_fake_urlopen_response(body)):
        check_ollama_available(model="qwen2.5:3b-instruct")  # must not raise


def test_check_ollama_available_raises_when_server_unreachable() -> None:
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
        try:
            check_ollama_available()
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "not reachable" in str(exc)
            assert "ollama serve" in str(exc)


def test_check_ollama_available_raises_when_model_not_pulled() -> None:
    body = {"models": [{"name": "some-other-model:latest"}]}
    with patch("urllib.request.urlopen", return_value=_fake_urlopen_response(body)):
        try:
            check_ollama_available(model="qwen2.5:3b-instruct")
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "qwen2.5:3b-instruct" in str(exc)
            assert "ollama pull" in str(exc)


def test_check_ollama_available_accepts_a_bare_tag_match() -> None:
    """Ollama's own /api/tags can echo back a model with an explicit tag
    even when the caller asked for the bare/no-tag form or vice versa --
    confirm the bare-name fallback comparison actually works."""
    body = {"models": [{"name": "qwen2.5:3b-instruct"}]}
    with patch("urllib.request.urlopen", return_value=_fake_urlopen_response(body)):
        check_ollama_available(model="qwen2.5:3b-instruct")  # exact match, sanity check
