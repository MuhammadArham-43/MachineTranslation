import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

# from model.Transformer import Transformer
from scratchImplementation.transformer import Transformer

from tqdm import tqdm
import os

from utils import translate_sentence
from hyperparameters import *


def tokenize_text(text, tokenizer):
    return [tok for tok in tokenizer(text)]


if __name__ == "__main__":

    spacy_ger = spacy.load('de_core_news_sm')
    spacy_eng = spacy.load('en_core_web_sm')

    german = Field(sequential=True, lower=True,
                   init_token='<sos>', eos_token='<eos>')
    english = Field(sequential=True, lower=True,
                    init_token='<sos>', eos_token='<eos>')

    train_data, valid_data, test_data = Multi30k.splits(
        exts=('.de', '.en'),
        fields=(german, english))

    english.build_vocab(train_data, max_size=10000, min_freq=2)
    german.build_vocab(train_data, max_size=10000, min_freq=2)

    src_vocab_size = len(german.vocab)
    tgt_vocab_size = len(english.vocab)

    SRC_PAD_IDX = english.vocab.stoi['<pad>']

    TENSORBAORD_DIR = 'runs/losses'
    SAVE_DIR = 'runs/models'
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(TENSORBAORD_DIR, exist_ok=True)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src)

    )

    # TRUE / False for Torch implementation. Only True for Scratch Implementation
    batch_first = True

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_pad_idx=SRC_PAD_IDX,
        embed_size=EMBEDDING_SIZE,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
        device=DEVICE,
        batch_first=batch_first
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=SRC_PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    test_sentence = 'ein pferd geht unter einer brücke neben einem boot.'

    print('TEST SENTENCE TRANSLATION')
    model.eval()
    print(translate_sentence(
        src_sentence=test_sentence,
        model=model,
        english=english,
        german=german,
        device=DEVICE,
    ))
    model.train()
    print()

    writer = SummaryWriter(log_dir=TENSORBAORD_DIR)
    step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for idx, batch in enumerate(tqdm(train_iterator)):
            input_data = batch.src.to(DEVICE)
            targets = batch.trg.to(DEVICE)

            # FOR BATCH FIRST IMPLEMENTATION
            if batch_first:
                input_data = input_data.permute(1, 0)
                targets = targets.permute(1, 0)

            if batch_first:
                output = model(input_data, targets[:, :-1])
            else:
                output = model(input_data, targets[:-1, :])

            # From (tgt_seq_len, N, tgt_vocab_size) to (tgt_seq_len * N, tgt_vocab_size)
            output = output.reshape(-1, output.shape[2])
            if batch_first:
                targets = targets[:, 1:].reshape(-1)
            else:
                targets = targets[1:, :].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            writer.add_scalar('Loss / Iteration', loss.item(), step)
            step += 1

        writer.add_scalar('Loss / Epoch', loss.item(), epoch)
        print(f'EPOCH {epoch}: Loss = {loss.item()}')
        print('TEST SENTENCE TRANSLATION')
        model.eval()
        print(translate_sentence(
            src_sentence=test_sentence,
            model=model,
            english=english,
            german=german,
            device=DEVICE,
        ))
        model.train()
        print()
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'latest.pth'))
