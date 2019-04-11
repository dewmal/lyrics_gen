import numpy as np
import torch
import torch.nn as nn
import spacy

from gen import Generator

nlp = spacy.load("en_core_web_sm")
print(nlp.vocab.vectors.data)
feature_size = 96

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_noise(batch_size, z_dim):
    return torch.randn((1, batch_size, feature_size), device=device)


def load_model():
    netG = Generator(1, feature_size).to(device)
    chkpt = torch.load("gan_model_2/_1_netG_37.pth")
    netG.load_state_dict(chkpt)
    netG.eval()
    return netG


def word_ids_to_sentence(sent_vects, vocab, join=None):
    """Converts a sequence of word ids to a sentence"""
    song = ''
    for sent_vec in sent_vects:
        sent = ''
        for word in sent_vec:
            print(word)
            sent = word + "\n"
            # sent += [vocab[ind] for ind in word]  # denumericalize
        song += sent
    return song


song_lyrics = load_model()(get_noise(17, 1))

print(song_lyrics.shape)

print(word_ids_to_sentence(song_lyrics.detach().cpu(), nlp.vocab, join=' '))
