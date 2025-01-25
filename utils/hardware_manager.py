import logging
import torch
from typing import Optional

class HardwareManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        self._current_device = None
        self.detect_hardware()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def detect_hardware(self) -> str:
        """Detect available hardware and set the appropriate device."""
        try:
            if torch.cuda.is_available():
                self._current_device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                self.logger.info(f"GPU detected: {gpu_name}")
                return 'cuda'
        except Exception as e:
            self.logger.warning(f"Error detecting GPU: {str(e)}")

        self._current_device = 'cpu'
        self.logger.info("Using CPU for computations")
        return 'cpu'

    @property
    def device(self) -> str:
        """Get the current device."""
        if self._current_device is None:
            self.detect_hardware()
        return self._current_device

    def get_device_object(self) -> torch.device:
        """Get the PyTorch device object."""
        return torch.device(self.device)

    def move_to_device(self, model: any) -> any:
        """Safely move a model to the current device with error handling."""
        try:
            return model.to(self.device)
        except Exception as e:
            self.logger.error(f"Error moving model to {self.device}: {str(e)}")
            if self.device == 'cuda':
                self.logger.info("Falling back to CPU")
                self._current_device = 'cpu'
                return model.to('cpu')
            raise e