from pathlib import Path
from typing import Optional

from transformers.trainer_callback import TrainerState


SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

class Vocabulary(object):
    def __init__(self, name):
        self.name = name
        self.token2count = {}
        self.index2token = [SOS_TOKEN, EOS_TOKEN]
        self.token2index = {token: idx for idx, token in enumerate(self.index2token)}

        self.sos_token_id = self.token2index[SOS_TOKEN]
        self.eos_token_id = self.token2index[EOS_TOKEN]

    @classmethod
    def from_index2token(self, index2token, sos_token=SOS_TOKEN, eos_token=EOS_TOKEN):
        vocab = Vocabulary("unknown")
        vocab.index2token = index2token
        vocab.token2index = {token: idx for idx, token in enumerate(vocab.index2token)}
        vocab.sos_token_id = vocab.token2index[sos_token]
        vocab.eos_token_id = vocab.token2index[eos_token]
        return vocab

    def add_token(self, token: str):
        if token not in self.token2index:
            self.token2index[token] = len(self.index2token)
            self.token2count[token] = 1
            self.index2token.append(token)
        else:
            self.token2count[token] += 1

    def add_sequence(self, sequence: list[str]):
        for token in sequence:
            self.add_token(token)

    def __len__(self):
        return len(self.index2token)

    def toJSON(self):
        return {
            "name": self.name,
            "index2token": self.index2token,
            "sos_token_id": self.sos_token_id,
            "eos_token_id": self.eos_token_id,
        }
    

def get_best_checkpoint(trainer_dir) -> Optional[str]:
    all_checkpoints = list(Path(trainer_dir).glob("checkpoint-*"))
    if not all_checkpoints:
        raise ValueError(f"No checkpoints found in {trainer_dir}")
    
    last_checkpoint = sorted(all_checkpoints, key=lambda p: int(p.name.split("-")[-1]))[-1]
    state = TrainerState.load_from_json(last_checkpoint / "trainer_state.json")
    best_checkpoint = state.best_model_checkpoint

    if best_checkpoint is None:
        # return the last one
        return str(last_checkpoint)

    return str(Path(trainer_dir) / Path(best_checkpoint).name)