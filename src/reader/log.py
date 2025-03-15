import sys

from loguru import logger

logger.remove()
logger.add(
    "reader.log",
    format="{message}",
    level="DEBUG",
    rotation="100 MB",
    compression="zip",
)

logger.add(sys.stdout, format="{message}", level="INFO")
