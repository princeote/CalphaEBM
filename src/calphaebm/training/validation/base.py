"""Base validator class."""

from abc import ABC, abstractmethod

import torch

from calphaebm.utils.logging import get_logger

logger = get_logger()


class BaseValidator(ABC):
    """Base class for all validators."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    @abstractmethod
    def validate(self, *args, **kwargs):
        """Run validation and return metrics."""
        pass
