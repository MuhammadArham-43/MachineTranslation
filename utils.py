import torch
from torch.nn import Module
import spacy
from torchtext.data.metrics import bleu_score
from torchtext.data import Field


def translate_sentence(
    src_sentence: str,
    model: Module,
    english: Field,
    german: Field,
    device: 'str',
    max_len: int = 50,
    batch_first=True
) -> str:
    spacy_ger = spacy.load('de_core_news_sm')
    tokens = [tok.text.lower() for tok in spacy_ger.tokenizer(src_sentence)]

    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    numericalized_sentence = [german.vocab.stoi[word] for word in tokens]

    if batch_first:
        src_tensor = torch.LongTensor(
            numericalized_sentence).unsqueeze(0).to(device)
    else:
        src_tensor = torch.LongTensor(
            numericalized_sentence).unsqueeze(1).to(device)

    outputs = [english.vocab.stoi[english.init_token]]
    for i in range(max_len):
        if batch_first:
            tgt_tensor = torch.LongTensor(outputs).unsqueeze(0).to(device)
        else:
            tgt_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            out = model(src_tensor, tgt_tensor)

        if batch_first:
            best_guess = out.argmax(2).squeeze(0)[-1].item()
        else:
            best_guess = out.argmax(2)[-1].item()

        outputs.append(best_guess)

        if best_guess == english.vocab.stoi[english.eos_token]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    return ' '.join(translated_sentence)
