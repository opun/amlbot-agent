import os
from dotenv import load_dotenv
load_dotenv()

# Expose key classes for easier imports
from .models import TracerConfig, TraceResult, Entity
from .tracer import CryptoTracer
from .mcp_client import MCPClient
