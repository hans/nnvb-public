"""
Utilities for working with transformer base models.
"""

from hydra.utils import instantiate
import transformers


def prepare_processor(config):
    if "facebook/wav2vec2" not in config.base_model.model_ref:
        raise NotImplementedError()

    tokenizer = instantiate(config.tokenizer)
    feature_extractor = instantiate(config.feature_extractor,
                                    padding_value=0.0,
                                    do_normalize=True,
                                    return_attention_mask=False)
    processor = transformers.Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor