from pathlib import Path

import datasets
import pyarrow as pa
import torch
import transformers

from beartype import beartype
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.datasets.speech_equivalence import SpeechHiddenStateDataset
from src.models.transformer import prepare_processor


@beartype
def extract_hidden_states(dataset: datasets.Dataset,
                          model: transformers.Wav2Vec2Model,
                          processor: transformers.Wav2Vec2Processor,
                          layer: int,
                          pseudo_causal: bool = False,
                          batch_size=12) -> SpeechHiddenStateDataset:
    flat_idxs = []
    frame_states_list = []
    compression_ratios = {}
    model.eval()

    def collate_batch(batch):
        batch = processor.pad(
            [{"input_values": values_i} for values_i in batch["input_values"]],
            max_length=None,
            return_tensors="pt",
            return_attention_mask=True)
        return batch

    def extract_representations(batch_items, idxs):
        batch = collate_batch(batch_items)

        with torch.no_grad():
            output = model(output_hidden_states=True,
                           input_values=batch["input_values"].to(model.device),
                           attention_mask=batch["attention_mask"].to(model.device))

        input_lengths = batch["attention_mask"].sum(dim=1)
        frame_lengths = model._get_feat_extract_output_lengths(input_lengths)

        # batch_size * sequence_length * hidden_size
        batch_hidden_states = output.hidden_states[layer].cpu()

        batch_compression_ratios = (frame_lengths / input_lengths.numpy())
        for idx, num_frames_i, hidden_states_i, compression_i in zip(idxs, frame_lengths, batch_hidden_states, batch_compression_ratios):
            flat_idxs.extend([(idx, j) for j in range(num_frames_i)])
            frame_states_list.append(hidden_states_i[:num_frames_i])
            compression_ratios[idx] = compression_i
    
    def extract_representations_pseudo_causal(item, idx, max_length=None, frame_counts=None):
        assert frame_counts is not None
        assert max_length is not None

        audio = torch.tensor(item["input_values"]).unsqueeze(0)
        audio_length = audio.shape[1]

        frame_counts = frame_counts[:audio_length]
        frame_keypoints = torch.nonzero(frame_counts.diff() > 0).squeeze(1) + 1

        attention_mask = torch.zeros(batch_size, audio_length, dtype=torch.int32).to(model.device)

        # NB we start from 1 here.
        # We want the output at frame i to have input sufficient for computing up and through
        # frame i. So we should map output frame i to input keypoint i+1.
        for i in range(1, frame_keypoints.shape[0], batch_size):
            batch_frame_targets = torch.arange(i, min(i + batch_size, frame_keypoints.shape[0]))
            batch_keypoints = frame_keypoints[i:i + batch_size]
            batch_length = max(batch_keypoints)
            real_batch_size = batch_keypoints.shape[0]
            
            batch_inputs = audio[:, :batch_length].to(model.device)
            batch_inputs = torch.tile(batch_inputs, (real_batch_size, 1))
            for j, frame_keypoint in enumerate(batch_keypoints):
                batch_inputs[j, frame_keypoint:] = 0
            
            attention_mask.fill_(0)
            for j, frame_keypoint in enumerate(batch_keypoints):
                attention_mask[j, :frame_keypoint] = 1

            with torch.no_grad():
                output = model(output_hidden_states=True,
                              input_values=batch_inputs,
                              attention_mask=attention_mask[:real_batch_size, :batch_length])

            batch_hidden_states = output.hidden_states[layer][torch.arange(real_batch_size), batch_frame_targets - 1].cpu()
            assert len(batch_hidden_states) == real_batch_size
            frame_states_list.append(batch_hidden_states)
            flat_idxs.extend([(idx, j - 1) for j in range(i, i + real_batch_size)])

        compression_ratios[idx] = frame_counts[-1].item() / audio_length

    # Extract and un-pad hidden representations from the model
    if pseudo_causal:
        max_length = max(pa.compute.list_value_length(dataset._data["input_values"]).to_pylist())

        # Find the input wav frames at which a new frame is created.
        # pre-calculate and share across the invocations of extract_representations_pseudo_causal
        frame_counts = model._get_feat_extract_output_lengths(torch.arange(0, max_length))

        dataset.map(extract_representations_pseudo_causal,
                    fn_kwargs={"frame_counts": frame_counts, "max_length": max_length},
                    batched=False,
                    with_indices=True,
                    desc="Extracting hidden states")
    else:
        dataset.map(extract_representations,
                    batched=True, batch_size=batch_size,
                    with_indices=True,
                    desc="Extracting hidden states")
    
    frame_states = torch.cat(frame_states_list, dim=0)
    frame_states = frame_states.unsqueeze(1).contiguous()
    # frame_states: total_num_frames * 1 * hidden_size

    return SpeechHiddenStateDataset(model.name_or_path, frame_states, compression_ratios, flat_idxs)


@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    processor = prepare_processor(config)
    dataset = datasets.load_from_disk(config.dataset.processed_data_dir).with_format("torch")
    
    model = transformers.Wav2Vec2Model.from_pretrained(config.base_model.model_ref).to(config.device)

    hidden_state_dataset = extract_hidden_states(
        dataset, model, processor, config.base_model.layer,
        pseudo_causal=config.base_model.pseudo_causal)
    
    hidden_state_dataset.to_hdf5(config.base_model.hidden_state_path)


if __name__ == "__main__":
    main()