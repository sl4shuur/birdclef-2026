"""The single mapping from configured encoder names to implementations."""

from birdclef.models.audio_cnn.config import EncoderName
from birdclef.models.audio_cnn.classifier_encoder import EfficientNetEncoder
from birdclef.models.encoder import ClassifierEncoder
from birdclef.models.perch.classifier_encoder import FrozenPerchEncoder


def encoder_class(name: EncoderName) -> type[ClassifierEncoder]:
    return {
        EncoderName.EFFICIENTNET: EfficientNetEncoder,
        EncoderName.PERCH: FrozenPerchEncoder,
    }[name]
