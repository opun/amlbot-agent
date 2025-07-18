import os
from dotenv import load_dotenv
load_dotenv()

import click
import logging
import sys
from .tracer import main as tracer_main

@click.command()
@click.option("-v", "--verbose", count=True)
def main(verbose: bool) -> None:
    """AMLBOT Tracer Agent - Agent for tracing simple cases"""
    logging_level = logging.WARN
    if verbose == 1:
        logging_level = logging.INFO
    elif verbose >= 2:
        logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level, stream=sys.stderr)
    
    # Run the tracer agent
    import asyncio
    asyncio.run(tracer_main())

if __name__ == "__main__":
    main()