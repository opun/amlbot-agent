import os


class ModelConfig:
    """Model configuration from environment with defaults."""

    ORCHESTRATOR_MODEL = os.getenv("OPENAI_ORCHESTRATOR_MODEL", "gpt-5-mini")
    SELECTOR_MODEL = os.getenv("OPENAI_SELECTOR_MODEL", "gpt-5-mini")
    VALIDATOR_MODEL = os.getenv("OPENAI_VALIDATOR_MODEL", "gpt-4o")
    JSON_RETRY_MODEL = os.getenv("OPENAI_JSON_RETRY_MODEL", "gpt-4o")
