from caikit.core.data_model import dataobject

@dataobject(
    package="text_sentiment.data_model",
    schema={
        "class_name": str,  # (required) Predicted relevant class name
        "confidence": float,  # (required) The confidence-like score of this prediction in [0, 1]
    },
)
class ClassInfo:
    """A single classification prediction."""

@dataobject(
    package="text_sentiment.data_model",
    schema={"classes": {"elements": ClassInfo}},
)
class ClassificationPrediction:
    """The result of a classification prediction."""

@dataobject(package="text_sentiment.data_model", schema={"text": str})
class TextInput:
    """A sample `domain primitive` input type for this library.
    The analog to a `Raw Document` for the `Natural Language Processing` domain."""


