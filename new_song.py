import numpy as np
import torch
import torch.nn as nn
import spacy
from numpy.linalg import norm

from gen import Generator

nlp = spacy.load("en_core_web_lg")
print(nlp.vocab.vectors.data)
feature_size = 96

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_noise(batch_size, z_dim):
    return torch.randn((1, batch_size, feature_size), device=device)


def load_model():
    netG = Generator(1, feature_size).to(device)
    chkpt = torch.load("models/gan_model_2/_1_netG_37.pth")
    netG.load_state_dict(chkpt)
    netG.eval()
    return netG


def most_similer_word(val):
    # cosine similarity
    cosine = lambda v1, v2: np.dot(v1, v2) / (norm(v1) * norm(v2))

    # gather all known words, take only the lowercased versions
    allWords = list({w for w in nlp.vocab})

    # sort by similarity to NASA
    allWords.sort(key=lambda w: cosine(w.vector, val))
    allWords.reverse()
    print("Top 10 most similar words to NASA:")
    for word in allWords[:10]:
        print(word.orth_)


def word_ids_to_sentence(sent_vects, vocab, join=None):
    """Converts a sequence of word ids to a sentence"""
    song = ''
    for sent_vec in sent_vects:
        sent = ''
        for words in sent_vec:
            for word in words:
                word = most_similer_word(word)
                print(word)
            # sent = word + "\n"
            # sent += [vocab[ind] for ind in word]  # denumericalize
        song += sent
    return song


song_lyrics = load_model()(get_noise(17, 1))

print(song_lyrics.shape)

print(most_similer_word(song_lyrics.detach().cpu()))
