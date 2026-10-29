"""EfficientNet inputs and module for the shared classifier."""

from birdclef.models.classifier.audio_cnn.network import AudioCNNModule
from birdclef.models.classifier.encoder import ClassifierEncoder
from birdclef.processing import log_mel


class EfficientNetEncoder(ClassifierEncoder):
    transform = staticmethod(log_mel)
    random_crop = True

    def module(self, *, initialize_pretrained: bool) -> AudioCNNModule:
        return AudioCNNModule(
            self.config.model,
            self.config.lr_scheduler,
            classes=self.classes,
            initialize_pretrained=initialize_pretrained,
        )
