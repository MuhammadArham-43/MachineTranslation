import torch
from utils import translate_sentence
from torchtext.data import Field
from torchtext.datasets import Multi30k
from model.Transformer import Transformer
from hyperparameters import *


if __name__ == "__main__":
    model_path = 'runs/models/latest.pth'

    german = Field(sequential=True, lower=True,
                   init_token='<sos>', eos_token='<eos>')
    english = Field(sequential=True, lower=True,
                    init_token='<sos>', eos_token='<eos>')

    train_data, valid_data, test_data = Multi30k.splits(
        exts=('.de', '.en'),
        fields=(german, english))

    english.build_vocab(train_data, max_size=10000, min_freq=2)
    german.build_vocab(train_data, max_size=10000, min_freq=2)

    model = Transformer(
        src_vocab_size=len(german.vocab),
        tgt_vocab_size=len(english.vocab),
        src_pad_idx=german.vocab.stoi['<pad>'],
        embed_size=EMBEDDING_SIZE,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        device=DEVICE
    )
    model.load_state_dict(torch.load(model_path))
    model.eval().to(DEVICE)

    german_sentence = 'ein pferd geht unter einer brücke neben einem boot.'
    translated_sentence = translate_sentence(
        src_sentence=german_sentence,
        model=model,
        english=english,
        german=german,
        device=DEVICE,
        max_len=50
    )
    print(translated_sentence)
