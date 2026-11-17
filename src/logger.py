"""Console logging for the pipeline. Import `logger` and use it like the standard library logger."""

import logging

logging.basicConfig(
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("retail_sales")
