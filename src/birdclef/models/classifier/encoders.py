"""The single mapping from configured encoder names to implementations."""

from birdclef.models.classifier.audio_cnn.classifier_encoder import EfficientNetEncoder
from birdclef.models.classifier.config import EncoderName
from birdclef.models.classifier.encoder import ClassifierEncoder
from birdclef.models.classifier.perch.classifier_encoder import FrozenPerchEncoder


def encoder_class(name: EncoderName) -> type[ClassifierEncoder]:
    return {
        EncoderName.EFFICIENTNET: EfficientNetEncoder,
        EncoderName.PERCH: FrozenPerchEncoder,
    }[name]
