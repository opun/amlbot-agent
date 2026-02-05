
import sys
import json
import logging
from datetime import datetime
from agent.models import TraceResult, CaseMeta, Entity, Path, Step, TraceStats
from agent.visualization import generate_visualization_payload

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def verify_payload_structure():
    # 1. Create Mock Data
    case_meta = CaseMeta(
        case_id="case-123",
        trace_id="trace-456",
        description="Test Case",
        victim_address="0xVictim",
        blockchain_name="eth",
        chains=["eth"],
        asset_symbol="ETH"
    )

    entities = [
        Entity(address="0xVictim", chain="eth", role="victim", labels=["Victim Wallet"]),
        Entity(address="0xAttacker", chain="eth", role="perpetrator", labels=["Attacker Wallet"]),
        Entity(address="0xService", chain="eth", role="cex_deposit", labels=["Binance"]),
    ]

    steps = [
        Step(
            step_index=0,
            from_address="0xVictim",
            to_address="0xAttacker",
            tx_hash="0xTx1",
            chain="trx",
            asset="USDT",
            amount_estimate=1.5,
            time=int(datetime.now().timestamp()),
            direction="outgoing",
            step_type="direct_transfer",
            reasoning="Theft"
        ),
        Step(
            step_index=1,
            from_address="0xAttacker",
            to_address="0xService",
            tx_hash="0xTx2",
            chain="trx",
            asset="TRX",
            amount_estimate=1.4,
            time=int(datetime.now().timestamp()),
            direction="outgoing",
            step_type="service_deposit",
            reasoning="Cashout"
        )
    ]

    paths = [
        Path(path_id="path-1", description="Main theft path", steps=steps)
    ]
    
    trace_stats = TraceStats(initial_amount_estimate=10.0, explored_paths=1)

    trace_result = TraceResult(
        case_meta=case_meta,
        entities=entities,
        paths=paths,
        annotations=[],
        trace_stats=trace_stats
    )

    # 2. Generate Payload
    payload_data = generate_visualization_payload(trace_result)
    
    # 3. Verify Structure
    logger.info("Verifying Top Level Fields...")
    assert "createdAt" in payload_data, "Missing createdAt"
    assert "thumbnail" in payload_data, "Missing thumbnail"
    assert "hash" in payload_data, "Missing hash"
    assert "payload" in payload_data, "Missing payload"
    assert "helpers" in payload_data, "Missing helpers"
    
    payload = payload_data["payload"]
    helpers = payload_data["helpers"]
    
    logger.info("Verifying Payload Fields...")
    assert "comments" in payload, "Missing payload.comments"
    assert "connects" in payload, "Missing payload.connects"
    assert "items" in payload, "Missing payload.items"
    assert "txs" in payload, "Missing payload.txs"
    assert "transform" in payload, "Missing payload.transform"
    
    # Verify Comments
    if payload["comments"]:
        comment = payload["comments"][0]
        logger.info(f"Checking Comment: {comment.get('descriptor')}")
        assert "width" in comment, "Comment missing width"
        assert "height" in comment, "Comment missing height"
        assert "isManuallyMoved" in comment, "Comment missing isManuallyMoved"
        assert "typeOfComment" in comment, "Comment missing typeOfComment"
        assert comment["typeOfComment"] == "comment", "Invalid typeOfComment"

    # Verify Connects
    if payload["connects"]:
        connect = payload["connects"][0]
        logger.info(f"Checking Connect: {connect.get('source')} -> {connect.get('target')}")
        data = connect.get("data", {})
        assert "color" in data, "Connect data missing color"
        # Optional: check specific fields if strictly required

    # Verify Items (Nodes)
    if payload["items"]:
        item = payload["items"][0]
        logger.info(f"Checking Item: {item.get('address')}")
        assert "extend" in item, "Item missing extend"
        extend = item["extend"]
        assert "currency" in extend, "Item extend missing currency"
        assert "token_id" in extend, "Item extend missing token_id"
        assert "owner" in extend, "Item extend missing owner"
        assert "type" in item, "Item missing type"
        assert item["type"] == "address", "Item type must be address"

    # Verify Txs
    if payload["txs"]:
        tx = payload["txs"][0]
        logger.info(f"Checking Tx: {tx.get('hash')}")
        assert "currency" in tx, "Tx missing currency"
        assert "type" in tx, "Tx missing type"
        assert tx["type"] == "txEth", "Tx type must be txEth"

    # Verify Helpers
    logger.info("Verifying Helpers...")
    assert "currencyInfo" in helpers, "Helpers missing currencyInfo"
    assert "txList" in helpers, "Helpers missing txList"
    
    # Verify Currency Info
    currencies = {c["currency"]: c for c in helpers["currencyInfo"]}
    assert "trx" in currencies, "Missing TRX currency info"
    # Check for USDT if present in trace (in mocks we used ETH, let's update mock to usage TRX for better test matching)
    
    # Verify AutoTxs logic (should be present even if empty list, but we want to test population)
    assert "autoTxs" in helpers, "Helpers missing autoTxs"

    if helpers["txList"]:
        tx_item = helpers["txList"][0]
        logger.info(f"Checking TxList Item: {tx_item.get('hash')}")
        assert "inputs" in tx_item, "TxList item missing inputs"
        assert "outputs" in tx_item, "TxList item missing outputs"
        assert "addressesCount" in tx_item, "TxList item missing addressesCount"
        assert "poolTime" in tx_item, "TxList item missing poolTime"
        
    logger.info("✅ Verification Successful!")
    # Print a snippet of currency info and autoTxs for manual check
    print("Currency Info:", json.dumps(helpers["currencyInfo"], indent=2))
    
    attacker_auto = next((item for item in helpers["autoTxs"] if item["address"] == "0xAttacker"), None)
    print("Attacker AutoTxs:", json.dumps(attacker_auto, indent=2))
    print("AutoTxs Sample:", json.dumps(helpers["autoTxs"][:1], indent=2))


if __name__ == "__main__":
    try:
        verify_payload_structure()
    except AssertionError as e:
        logger.error(f"❌ Verification Failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)
