from .frustum_assigner import FrustumAssigner
from .point_assigner import PointInBoxAssigner
from .dist_assigner import DistAssigner
from .hybrid_assigner import HybridAssigner
from .ota_assigner import OTAAssigner

__all__ = [
    'FrustumAssigner', 'PointInBoxAssigner', 'DistAssigner', 'HybridAssigner',
    'OTAAssigner',
]