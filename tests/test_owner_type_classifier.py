"""Tests for _classify_by_owner_type — the structural (type-based)
classifier that runs before the keyword heuristic.

Real-world failure that motivated this: Tronify on TRON has
``owner={name: "Tronify", slug: "Tronify", type: "other", subtype: null}``.
None of the name/slug strings hit a bridge/dex/exchange keyword, so the
old keyword heuristic returned ``intermediate`` and the tracer stepped
straight through. The structural check catches this via ``type == "other"``.
"""
from __future__ import annotations

import pytest

from agent.base_tracer import BaseTracer


classify = BaseTracer._classify_by_owner_type


class TestStructuralClassifier:
    def test_exchange_unlicensed_is_cex_terminal(self):
        out = classify({"name": "Arvix.pro(abcex.io)", "slug": "Arvix",
                        "type": "exchange_unlicensed", "subtype": None})
        assert out == {
            "role": "cex_deposit", "terminal": True,
            "service_label": "Arvix.pro(abcex.io)", "protocol": None,
        }

    def test_exchange_licensed_is_cex_terminal(self):
        out = classify({"name": "Binance", "slug": "binance",
                        "type": "exchange_licensed", "subtype": None})
        assert out["role"] == "cex_deposit"
        assert out["terminal"] is True

    def test_plain_exchange_type_is_cex_terminal(self):
        out = classify({"name": "SomeCEX", "type": "exchange"})
        assert out["role"] == "cex_deposit"
        assert out["terminal"] is True

    def test_p2p_dex_is_dex_terminal(self):
        out = classify({"name": "SunSwap", "slug": "SunSwap",
                        "type": "p2p_exchange_unlicensed", "subtype": "DEX"})
        assert out["role"] == "dex_service"
        assert out["terminal"] is True
        assert out["service_label"] == "SunSwap"

    def test_p2p_without_dex_subtype_is_otc_terminal(self):
        out = classify({"name": "LocalPeer", "type": "p2p_exchange_unlicensed", "subtype": None})
        assert out["role"] == "otc_service"
        assert out["terminal"] is True

    def test_bridge_type_is_terminal(self):
        out = classify({"name": "Stargate", "slug": "stargate", "type": "bridge"})
        assert out["role"] == "bridge_service"
        assert out["terminal"] is True

    def test_mixer_is_terminal(self):
        out = classify({"name": "TornadoPool", "type": "mixer"})
        assert out["role"] == "unidentified_service"
        assert out["terminal"] is True
        assert out["service_label"] == "Mixer"

    def test_other_with_name_is_identified_service_terminal(self):
        """Tronify — the case that prompted this whole commit."""
        out = classify({"name": "Tronify", "slug": "Tronify",
                        "type": "other", "subtype": None})
        assert out["role"] == "unidentified_service"
        assert out["terminal"] is True
        assert out["service_label"] == "Tronify"

    def test_other_with_bridge_subtype_becomes_bridge_service(self):
        """Bridgers on ETH: {type: other, subtype: Bridge} — must classify
        as bridge_service so the tracer triggers bridge_analyze and
        continues on the destination chain rather than stopping at the
        bridge contract."""
        out = classify({"name": "Bridgers", "slug": "Bridgers",
                        "type": "other", "subtype": "Bridge"})
        assert out["role"] == "bridge_service"
        assert out["terminal"] is True
        assert out["service_label"] == "Bridgers"

    def test_other_with_dex_subtype_becomes_dex(self):
        out = classify({"name": "SomePool", "type": "other", "subtype": "DEX"})
        assert out["role"] == "dex_service"
        assert out["terminal"] is True

    def test_other_with_known_bridge_brand_name(self):
        """LayerZero returned as type=other with no subtype — brand
        allowlist catches it."""
        for brand in ("LayerZero", "Stargate", "Wormhole", "Symbiosis", "Mayan", "Allbridge"):
            out = classify({"name": brand, "slug": brand,
                            "type": "other", "subtype": None})
            assert out is not None, f"{brand} should be classified"
            assert out["role"] == "bridge_service", f"{brand} → {out['role']}"
            assert out["terminal"] is True

    def test_other_with_miner_subtype_is_terminal_unidentified(self):
        """ViaBTC mining pool — stop, don't chase block rewards."""
        out = classify({"name": "ViaBTC", "slug": "ViaBTC",
                        "type": "other", "subtype": "miner"})
        assert out["role"] == "unidentified_service"
        assert out["terminal"] is True
        assert out["service_label"] == "ViaBTC"

    def test_other_without_name_falls_through(self):
        assert classify({"name": None, "slug": None, "type": "other"}) is None

    def test_stolen_coins_is_not_terminal(self):
        """Community victim-report markers (``owner.type='stolen_coins'``)
        tell us the funds being traced are dirty — NOT that we've reached
        a destination. Classifier must keep tracing. Regression: before
        the fix, LLM was inferring role=victim terminal=true from the
        type alone, cutting off traces at TBdxz (Victim report #16547)
        when the money was actually flowing further through that mule."""
        out = classify({
            "name": "Victim report #16547", "slug": "Victim-report-16547",
            "type": "stolen_coins", "subtype": "Community Report",
        })
        assert out is not None
        assert out["role"] == "intermediate"
        assert out["terminal"] is False
        assert out["service_label"] == "Stolen funds"

    def test_stolen_coins_does_not_hit_exchange_prefix(self):
        """Regression guard: ``stolen_coins`` must NOT trip the
        ``owner_type.startswith('exchange')`` branch or similar — the
        explicit rule runs first."""
        out = classify({"name": "X", "type": "stolen_coins"})
        assert out["role"] == "intermediate"
        assert out["terminal"] is False

    def test_unknown_name_falls_through(self):
        # Guarding against API quirks where name is literally "unknown".
        assert classify({"name": "unknown", "type": "other"}) is None

    def test_unrecognized_type_falls_through(self):
        assert classify({"name": "Foo", "type": "wallet"}) is None

    def test_none_owner_returns_none(self):
        assert classify(None) is None

    def test_non_dict_owner_returns_none(self):
        assert classify("Binance") is None


class TestHeuristicDelegatesToStructural:
    """_heuristic_classify must prefer structural over keyword when both fire."""

    def test_delegates_tronify_to_structural(self):
        tracer = _make_tracer()
        owner = {"name": "Tronify", "slug": "Tronify", "type": "other"}
        out = tracer._heuristic_classify(owner=owner, services={}, owner_hint=None)
        assert out["terminal"] is True  # was False before the fix
        assert out["role"] == "unidentified_service"
        assert out["service_label"] == "Tronify"

    def test_delegates_exchange_via_type_not_name(self):
        """An exchange whose *name* contains no keyword still terminates."""
        tracer = _make_tracer()
        owner = {"name": "Обменник №7", "slug": "obmen-7",
                 "type": "exchange_unlicensed"}
        out = tracer._heuristic_classify(owner=owner, services={}, owner_hint=None)
        assert out["role"] == "cex_deposit"
        assert out["terminal"] is True

    def test_fallback_to_keyword_when_type_unknown(self):
        """No type, but name="Binance" — keyword heuristic still wins."""
        tracer = _make_tracer()
        owner = {"name": "Binance", "slug": "binance"}
        out = tracer._heuristic_classify(owner=owner, services={}, owner_hint=None)
        assert out["role"] == "cex_deposit"
        assert out["terminal"] is True

    def test_null_owner_falls_through(self):
        tracer = _make_tracer()
        out = tracer._heuristic_classify(owner=None, services={}, owner_hint=None)
        assert out["role"] == "intermediate"
        assert out["terminal"] is False


def _make_tracer() -> BaseTracer:
    """Instantiate a minimal concrete BaseTracer for pure-method tests.

    BaseTracer.__init__ spins up an httpx + AsyncOpenAI client. That's
    fine — we never make a real network call here, we only touch
    ``_classify_by_owner_type`` / ``_heuristic_classify`` which are pure.
    """
    class _T(BaseTracer):
        async def execute_tool(self, tool_name, arguments):
            raise NotImplementedError

        def _get_client(self):
            raise NotImplementedError
    return _T()
