# Modified form of the librispeech_asr dataset from huggingface
# with preprocessed resampled/wav-converted audio files
# and integrating TextGrid word/phone-level annotations

# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Librispeech automatic speech recognition dataset."""


import re
import os
from pathlib import Path

import datasets
from datasets.tasks import AutomaticSpeechRecognition
import textgrid


_CITATION = """\
@inproceedings{panayotov2015librispeech,
  title={Librispeech: an ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on},
  pages={5206--5210},
  year={2015},
  organization={IEEE}
}
"""

_DESCRIPTION = """\
LibriSpeech is a corpus of approximately 1000 hours of read English speech with sampling rate of 16 kHz,
prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read
audiobooks from the LibriVox project, and has been carefully segmented and aligned.87
"""

class LibrispeechASRConfig(datasets.BuilderConfig):
    """BuilderConfig for LibriSpeechASR."""

    def __init__(self, alignment_dir, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        super(LibrispeechASRConfig, self).__init__(version=datasets.Version("2.1.0", ""), **kwargs)
        self.alignment_dir = alignment_dir
        self.audio_sample_rate = 16000


class LibrispeechASR(datasets.GeneratorBasedBuilder):
    """Librispeech dataset."""

    DEFAULT_WRITER_BATCH_SIZE = 256
    BUILDER_CONFIG_CLASS = LibrispeechASRConfig
    # DEFAULT_CONFIG_NAME = "all"
    # BUILDER_CONFIGS = [
    #     LibrispeechASRConfig(name="clean", description="'Clean' speech."),
    #     LibrispeechASRConfig(name="other", description="'Other', more challenging, speech."),
    #     LibrispeechASRConfig(name="all", description="Combined clean and other dataset."),
    # ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),

                    # passing null / false here because the data is already preprocessed
                    "audio": datasets.Audio(sampling_rate=None, mono=False),

                    "text": datasets.Value("string"),
                    "speaker_id": datasets.Value("int64"),
                    "chapter_id": datasets.Value("int64"),
                    "id": datasets.Value("string"),

                    "word_detail": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "stop": datasets.Value("int64"),
                            "utterance": datasets.Value("string"),
                        }
                    ),
                    "phonetic_detail": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "stop": datasets.Value("int64"),
                            "utterance": datasets.Value("string"),
                        }
                    ),
                }
            ),
            supervised_keys=("file", "text"),
            citation=_CITATION,
            task_templates=[AutomaticSpeechRecognition(audio_column="audio", transcription_column="text")],
        )

    def _split_generators(self, dl_manager):
        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"{data_dir} does not exist. Make sure you insert a manual dir via `datasets.load_dataset('librispeech', data_dir=...)`"
            )
        
        split_name = re.sub(r"[^\w]+", ".", os.path.basename(data_dir))
        num_examples = len(list(Path(data_dir).glob(f"**/*.wav")))
        gen = datasets.SplitGenerator(
            name=split_name,
            gen_kwargs={"split_dir": os.path.basename(data_dir), "data_dir": data_dir})
        gen.split_info.num_examples = num_examples

        return [gen]

    def _generate_examples(self, data_dir, split_dir):
        """Generate examples from a LibriSpeech data_dir."""
    
        key = 0
        audio_data = {}
        for chapter_path in Path(data_dir).glob(f"LibriSpeech/{split_dir}/*/*"):
            speaker_id = chapter_path.parent.name
            chapter_id = chapter_path.name

            # Load transcript
            transcripts = {}
            with (chapter_path / f"{speaker_id}-{chapter_id}.trans.txt").open() as transcript_f:
                for line in transcript_f:
                    if line:
                        line = line.strip()
                        id_, transcript = line.split(" ", 1)
                        transcripts[id_] = transcript

            # Load alignments
            alignments = {}
            alignment_dir = Path(self.config.alignment_dir) / split_dir / speaker_id / chapter_id
            for alignment_path in alignment_dir.glob("*.TextGrid"):
                full_id = alignment_path.stem
                tg = textgrid.TextGrid.fromFile(str(alignment_path))

                alignments[full_id] = {}
                for tier_name, target_name in [("words", "word_detail"), ("phones", "phonetic_detail")]:
                    tier = tg.getFirst(tier_name)
                    starts, stops, utterances = [], [], []
                    for interval in tier:
                        starts.append(int(interval.minTime * self.config.audio_sample_rate))
                        stops.append(int(interval.maxTime * self.config.audio_sample_rate))
                        utterances.append(interval.mark)

                    alignments[full_id][target_name] = {"start": starts, "stop": stops, "utterance": utterances}

            for audio_path in chapter_path.glob("*.wav"):
                full_id = audio_path.stem
                if full_id not in alignments:
                    continue

                yield full_id, {
                    "id": full_id,
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "file": str(audio_path),
                    "text": transcripts[full_id],

                    "audio": {
                        "path": str(audio_path),
                        "bytes": audio_path.read_bytes(),
                    },

                    **alignments[full_id],
                }