"""Regression: per-swap bridge deposit addresses must be classified as
the bridge service itself, not as plain intermediates.

Background — TWO DIFFERENT BRIDGE PATTERNS:

  Pattern A: NEAR Intents per-swap deposit (also covers NEAR Omni
  Bridge / NEAR One — same protocol under multiple display names).

    victim → perpetrator → … → ``TShMgnKsc…`` (deposit, no owner tag)
                                    └──→ ``TX5XiRX…`` ("NEAR Intents Treasury")

    The deposit address is bridge-managed (one-shot, per-order). The
    bridge tx (from the user's perspective) IS the *incoming* tx that
    funded the deposit. ``bridge_analyze`` on the funding tx returns
    ``is_bridge=true`` with a destination on the dst_chain.

  Pattern B: User wallet uses a bridge (Bridgers, etc.).

    victim → perpetrator wallet → ┬─→ Bridgers contract (tx1, swap A)
                                  ├─→ Bridgers contract (tx2, swap B)
                                  └─→ unrelated address

    The wallet itself is the user's, NOT a bridge address. The bridge
    tx is the *outgoing* tx to the Bridgers contract. ``bridge_analyze``
    on the *incoming* tx (which is just a regular deposit/transfer)
    returns ``is_bridge=false``. The existing per-hop handler will
    then catch the Bridgers contract on the NEXT hop and follow the
    cross-chain leg from there.

``_detect_bridge_deposit_pattern`` is a SIGNAL — it spots both A and B
because they share the same outflow shape (≥70% to a bridge brand).
The call site disambiguates by re-running ``bridge_analyze`` on the
funding tx:

  * ``is_bridge=true`` → confirmed Pattern A, reclassify as
    ``bridge_service`` and follow the destination.
  * ``is_bridge=false`` → Pattern B (or false positive), fall through
    silently to normal outflow processing. The existing per-hop bridge
    handler picks up the Bridgers contract one hop later.
"""
from __future__ import annotations

import pytest

from agent.base_tracer import BaseTracer


class _T(BaseTracer):
    async def execute_tool(self, tool_name, arguments):
        raise NotImplementedError

    def _get_client(self):
        raise NotImplementedError


@pytest.fixture
def tracer():
    return _T()


# ---- _BRIDGE_BRAND_NAMES + _classify_by_owner_type ----

class TestNearIntentsBrandClassification:
    """The treasury aggregator address (``TX5XiRX…``) must classify as
    ``bridge_service`` so the existing bridge-handling fallback fires
    when the trace reaches it directly (e.g. a different perp uses it
    via a path that bypasses the deposit-detection heuristic)."""

    def test_near_intents_treasury_classifies_as_bridge(self):
        owner = {
            "name": "NEAR Intents Treasury",
            "slug": "NEAR-Intents-Treasury",
            "type": "other",
            "subtype": None,
        }
        result = BaseTracer._classify_by_owner_type(owner)
        assert result is not None
        assert result["role"] == "bridge_service"
        assert result["terminal"] is True
        assert result["service_label"] == "NEAR Intents Treasury"

    def test_near_omni_bridge_classifies_as_bridge(self):
        owner = {
            "name": "NEAR Omni Bridge",
            "slug": "NEAR-Omni-Bridge",
            "type": "other",
            "subtype": None,
        }
        result = BaseTracer._classify_by_owner_type(owner)
        assert result is not None
        assert result["role"] == "bridge_service"

    def test_random_other_owner_does_not_classify_as_bridge(self):
        """No false positive on unrelated ``type=other`` owners."""
        owner = {
            "name": "Some Random Service",
            "slug": "some-random",
            "type": "other",
            "subtype": None,
        }
        result = BaseTracer._classify_by_owner_type(owner)
        if result is not None:
            assert result["role"] != "bridge_service"


class TestTokenContractClassifiesAsBridge:
    """LayerZero OFT contracts (USDT0 etc.) come back as
    ``{type: "other", subtype: "ERC/BEP-20 Token Contract"}``. They're
    not opaque "unidentified services" — sending tokens to them
    triggers a cross-chain mint/burn. Classify as ``bridge_service``
    so the existing bridge handler runs ``bridge_analyze`` on the
    incoming tx and follows any cross-chain destination."""

    def test_layerzero_oft_usdt0_classifies_as_bridge(self):
        owner = {
            "name": "USDT0",
            "slug": "USDT0",
            "type": "other",
            "subtype": "ERC/BEP-20 Token Contract",
        }
        result = BaseTracer._classify_by_owner_type(owner)
        assert result is not None
        assert result["role"] == "bridge_service"
        assert result["terminal"] is True
        assert result["service_label"] == "USDT0"

    def test_trc20_token_contract_also_classifies_as_bridge(self):
        """Subtype matching is substring on lowercase form, so any
        ``*Token Contract*`` variant fires (TRC-20, BEP-20, ERC-20,
        future chain-specific variants)."""
        for subtype in (
            "TRC-20 Token Contract",
            "ERC-20 Token Contract",
            "BEP-20 Token Contract",
            "Solana Token Contract",
        ):
            owner = {
                "name": "SomeOFT",
                "slug": "SomeOFT",
                "type": "other",
                "subtype": subtype,
            }
            result = BaseTracer._classify_by_owner_type(owner)
            assert result is not None, f"no classification for subtype={subtype!r}"
            assert result["role"] == "bridge_service", (
                f"subtype={subtype!r} must classify as bridge_service "
                f"so bridge_analyze can resolve cross-chain destination; "
                f"got {result}"
            )

    def test_token_contract_subtype_takes_priority_over_brand_fallback(self):
        """Token-contract detection must run BEFORE the brand-name
        fallback so we don't depend on brand presence to catch OFTs.
        ``"USDT0"`` IS in ``_BRIDGE_BRAND_NAMES`` (defense in depth),
        but the structural subtype check should also cover unknown
        OFT brands the registry doesn't know about yet."""
        owner = {
            "name": "ZZUnknownOFT",  # NOT in brand allowlist
            "slug": "zz-unknown-oft",
            "type": "other",
            "subtype": "ERC-20 Token Contract",
        }
        result = BaseTracer._classify_by_owner_type(owner)
        assert result is not None
        assert result["role"] == "bridge_service"
        assert result["service_label"] == "ZZUnknownOFT"

    def test_usdt0_listed_in_bridge_brand_allowlist(self):
        """Defense in depth: even if the API drops the
        ``Token Contract`` subtype, ``USDT0`` should still be picked
        up by the brand-name fallback."""
        assert "usdt0" in BaseTracer._BRIDGE_BRAND_NAMES
        assert "usdc0" in BaseTracer._BRIDGE_BRAND_NAMES
        assert "oft" in BaseTracer._BRIDGE_BRAND_NAMES


# ---- _owner_matches_bridge_brand ----

class TestOwnerMatchesBridgeBrand:
    def test_matches_near_intents_by_name(self):
        owner = {"name": "NEAR Intents Treasury", "slug": "NEAR-Intents-Treasury"}
        assert BaseTracer._owner_matches_bridge_brand(owner) is not None

    def test_matches_bridgers_by_slug(self):
        owner = {"name": "Bridgers", "slug": "Bridgers"}
        assert BaseTracer._owner_matches_bridge_brand(owner) == "bridgers"

    def test_no_match_on_blank_owner(self):
        assert BaseTracer._owner_matches_bridge_brand(None) is None
        assert BaseTracer._owner_matches_bridge_brand({}) is None
        assert BaseTracer._owner_matches_bridge_brand({"name": "", "slug": ""}) is None

    def test_no_match_on_unrelated_owner(self):
        owner = {"name": "Binance", "slug": "Binance"}
        assert BaseTracer._owner_matches_bridge_brand(owner) is None


# ---- _detect_bridge_deposit_pattern ----

NEAR_TREASURY_OWNER = {
    "global_id": 97299,
    "id": 331775336,
    "name": "NEAR Intents Treasury",
    "slug": "NEAR-Intents-Treasury",
    "type": "other",
    "subtype": None,
}


class TestDetectBridgeDepositPattern:
    def test_detects_when_only_outflow_goes_to_bridge_brand(self, tracer):
        """The TShMgn / TRSAwXt case: deposit address with ONE outflow,
        going to the NEAR Intents Treasury aggregator."""
        data_list = [
            {
                "hash": "f5b894a1bd11220ac468bb6bec1daafc3882b550f146876ad922b0bd18aa3909",
                "amount_coerced": 500000,
                "counterparty": [NEAR_TREASURY_OWNER],
                "token_id": 9,
            }
        ]
        owner = tracer._detect_bridge_deposit_pattern(data_list)
        assert owner is not None
        assert owner["name"] == "NEAR Intents Treasury"

    def test_no_detection_when_no_brand_outflow(self, tracer):
        """Normal wallet sending to an unrelated address — must NOT
        trigger reclassification."""
        data_list = [
            {
                "hash": "abc",
                "amount_coerced": 100000,
                "counterparty": [{"name": "Random Wallet", "slug": "random"}],
                "token_id": 9,
            }
        ]
        assert tracer._detect_bridge_deposit_pattern(data_list) is None

    def test_no_detection_below_threshold(self, tracer):
        """30% to bridge, 70% elsewhere — below the 70% dominant-share
        threshold, so this is a regular wallet that happens to use a
        bridge for a small share, not a dedicated deposit address."""
        data_list = [
            {
                "hash": "tx1",
                "amount_coerced": 30,
                "counterparty": [NEAR_TREASURY_OWNER],
            },
            {
                "hash": "tx2",
                "amount_coerced": 70,
                "counterparty": [{"name": "Random", "slug": "random"}],
            },
        ]
        assert tracer._detect_bridge_deposit_pattern(data_list) is None

    def test_detection_fires_at_threshold(self, tracer):
        """≥ 70% to bridge brand triggers reclassification."""
        data_list = [
            {
                "hash": "tx1",
                "amount_coerced": 80,
                "counterparty": [NEAR_TREASURY_OWNER],
            },
            {
                "hash": "tx2",
                "amount_coerced": 20,
                "counterparty": [{"name": "Random", "slug": "random"}],
            },
        ]
        owner = tracer._detect_bridge_deposit_pattern(data_list)
        assert owner is not None
        assert owner["name"] == "NEAR Intents Treasury"

    def test_handles_output_owner_field(self, tracer):
        """``output_owner`` is the alternative shape some endpoints
        return instead of ``counterparty[0]``."""
        data_list = [
            {
                "hash": "tx1",
                "amount_coerced": 1000,
                "output_owner": NEAR_TREASURY_OWNER,
            }
        ]
        assert tracer._detect_bridge_deposit_pattern(data_list) is not None

    def test_skips_zero_amount_outflows(self, tracer):
        """Zero-amount approvals/dust must not skew the volume share."""
        data_list = [
            {
                "hash": "approval",
                "amount_coerced": 0,
                "counterparty": [{"name": "Random", "slug": "random"}],
            },
            {
                "hash": "real",
                "amount_coerced": 1000,
                "counterparty": [NEAR_TREASURY_OWNER],
            },
        ]
        assert tracer._detect_bridge_deposit_pattern(data_list) is not None

    def test_empty_list_returns_none(self, tracer):
        assert tracer._detect_bridge_deposit_pattern([]) is None
        assert tracer._detect_bridge_deposit_pattern(None) is None  # type: ignore[arg-type]

    def test_pattern_b_user_wallet_using_bridgers_also_detects(self, tracer):
        """The detection is a SIGNAL — it can't tell Pattern A apart
        from Pattern B by outflow shape alone (both have ≥ 70% volume
        going to a bridge brand). Reproduces the
        ``0xe0c92b55…`` / ``0x0bd2af6a…`` Bridgers case where the
        perpetrator's wallet had four outflows: one ordinary 0.0077 ETH
        transfer + three Bridgers swaps totalling 0.2834 ETH (97.4% to
        Bridgers).

        The call site is responsible for disambiguating with
        ``bridge_analyze`` on the funding tx — when that returns
        ``is_bridge=false`` the trace MUST fall through to normal
        outflow processing instead of reclassifying the wallet.
        """
        bridgers_owner = {
            "global_id": 86455,
            "id": 346809816,
            "name": "Bridgers",
            "slug": "Bridgers",
        }
        data_list = [
            {
                "hash": "0x4e4a9cec",
                "amount_coerced": 0.0077,
                # No counterparty owner — unrelated outflow.
            },
            {
                "hash": "0xe4a4af5f",
                "amount_coerced": 0.1418735163,
                "counterparty": [bridgers_owner],
            },
            {
                "hash": "0xc1dc9c6f",
                "amount_coerced": 0.1380876394,
                "counterparty": [bridgers_owner],
            },
            {
                "hash": "0xffde1407",
                "amount_coerced": 0.0036,
                "counterparty": [bridgers_owner],
            },
        ]
        # Detection MUST fire — outflow share to Bridgers is ~97%.
        owner = tracer._detect_bridge_deposit_pattern(data_list)
        assert owner is not None
        assert owner["name"] == "Bridgers"
