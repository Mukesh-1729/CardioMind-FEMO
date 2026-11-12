import logging
from ..data.partition import NonIIDPartitioner

logger = logging.getLogger(__name__)

# Re-export for federated module access
__all__ = ['NonIIDPartitioner']
