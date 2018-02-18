import numpy as np
from collections import defaultdict, Counter
import pickle
import string
import pprint
import torch
from torch.autograd import Variable
import re


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
ROOT_TOKEN = '<root>'
RANDOM_SEED = 1


def parse_conllu(filename, clean=True):
    """
    Parse a .conllu file to a list of sentences. Each sentence is a 2d list with each row
    a word, and with columns 'idx', 'word', 'POS tag', 'arc head', 'arc label'.
    Args:
        filename: string, file to parse
        clean: boolean, if True, remove sentences with arcs that contain underscores.

    Returns: List of sentences

    """
    cols = [0, 1, 3, 6, 7]
    with open(filename, 'r', encoding='utf-8') as f:
        # read all lines, remove comments
        data = [line for line in f.readlines() if not line.startswith('#')]
    
    # split sentences
    newline_idx = [i for i, s in enumerate(data) if s == '\n']
    sentences = []
    prev_split = 0
    for split in newline_idx:
        sentences.append(data[prev_split:split])
        prev_split = split + 1
    
    # select useful cols
    for i, s in enumerate(sentences):
        s = np.array([word.strip().split('\t') for word in s])
        sentences[i] = s[:, cols]
    
    # remove sentences with words without head
    if clean:
        sentences = [s for s in sentences if '_' not in s[:,4]]

    return sentences


def filter_words(sentences, filter_single=True):
    """
    Applies a series of filter to each word in each sentence. Filters
    are applied in this order:
    - replace urls with an <url> tag.
    - replace a string of more than 2 punctuations with a <punct> tag.
    - replace strings that contain digits with a <num> tag.
    - if filter_single, replace words that only occur once with UNK_TOKEN.
      This step is useful when parsing training data, to make sure the UNK_TOKEN
      in the word embeddings gets trained.

    Args:
        sentences: list of sentences, from parse_conllu.
        filter_single: boolean, if true replace words that occur once with UNK_TOKEN.

    Returns: List of sentences with words filtered.
    """
    filtered = []
    word_counts = get_word_counts(sentences)
    one_words = set([w for w, c in word_counts.items() if c==1])
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            if is_url(word[1]):
                sentence[j, 1] = '<url>'
            elif is_long_punctuation(word[1]):
                sentence[j, 1] = '<punct>'
            elif has_digits(word[1]):
                sentence[j, 1] = '<num>'
            elif filter_single and word[1].lower() in one_words:
                sentence[j, 1] = UNK_TOKEN
        filtered.append(sentence)

    return filtered

def get_word_counts(sentences):
    """
    Create a Counter of all words in sentences, in lowercase.
    Args:
        sentences: List of sentences, from parse_conllu.

    Returns: Counter with word: count.

    """
    words = [word[1].lower() for sentence in sentences for word in sentence]
    return Counter(words)


def is_url(word):
    """
    Lazy check if a word is an url. True if word contains all of {':' '/' '.'}.
    """
    return bool(set('./:').issubset(word))


def is_long_punctuation(word):
    """
    True if word is longer than 2 and only contains interpunction.
    """
    return bool(len(word) > 2 and set(string.punctuation).issuperset(word))


def has_digits(word):
    """
    True if word contains digits.
    """
    return bool(set(string.digits).intersection(word))


def get_index_mappings(sentences):
    """
    Create an index mapping of each word, POS tag and arc label in sentences.
    example use:
    idx = x2i['word']['apple']
    word = i2x['word'][idx]

    Args:
        sentences: list of sentences, from parse_conllu

    Returns: dictionaries x2i and i2x, which contain the translation to and from indices.

    """
    # instantiate dicts
    w2i = defaultdict(lambda: len(w2i))
    i2w = dict()
    t2i = defaultdict(lambda: len(t2i))
    i2t = dict()
    l2i = defaultdict(lambda: len(l2i))
    i2l = dict()

    # Default values
    i2w[w2i[PAD_TOKEN]] = PAD_TOKEN
    i2w[w2i[UNK_TOKEN]] = UNK_TOKEN
    i2w[w2i[ROOT_TOKEN]] = ROOT_TOKEN

    i2t[t2i[PAD_TOKEN]] = PAD_TOKEN
    i2t[t2i[UNK_TOKEN]] = UNK_TOKEN
    i2t[t2i[ROOT_TOKEN]] = ROOT_TOKEN

    # Fill dicts
    words = set()
    tags = set()
    labels = set()
    for sentence in sentences:
        for word_array in sentence:
            words.add(word_array[1].lower())
            labels.add(word_array[4])
            tags.add(word_array[2])

    for word in sorted(list(words)):
        i2w[w2i[word]] = word
    for tag in sorted(list(tags)):
        i2t[t2i[tag]] = tag
    for label in sorted(list(labels)):
        i2l[l2i[label]] = label

    # collect dicts
    x2i = {"word":dict(w2i), "tag":dict(t2i), "label":dict(l2i)}
    i2x = {"word":dict(i2w), "tag":dict(i2t), "label":dict(i2l)}
    return x2i, i2x


def tokenize_sentences(sentences, x2i):
    """
    Convert each sentence to int arrays using mappings in x2i.
    """

    w2i = x2i['word']
    t2i = x2i['tag']
    l2i = x2i['label']
    sentences_idx = []
    for s in sentences:
        s_idx = []
        s_idx.append([0, w2i[ROOT_TOKEN], t2i[ROOT_TOKEN], -1, -1])
        for i, si in enumerate(s):
            word_idx = w2i.get(si[1].lower(), w2i[UNK_TOKEN])
            tag_idx = t2i.get(si[2], t2i[UNK_TOKEN])
            lab_idx = l2i[si[4]]
            s_idx.append([int(si[0]), word_idx, tag_idx, int(si[3]), lab_idx])
        sentences_idx.append(np.vstack(s_idx).astype(int))
    return sentences_idx


def make_dataset(filename, vocab_file=None, train_phase=True, return_sentences=False):
    """ 
    Parses conllu file to list of sentences, apply filters in filter_words,
    generates word, tag and label vocabularies and index mappings, and
    converts all sentences to int arrays. If vocab_file is given, use those vocabs instead.
    if return_words, return raw strings (without filters) from conllu file.
    """
    sentences = parse_conllu(filename)

    sentences = filter_words(sentences, filter_single=train_phase)

    # Load/make vocabs
    if vocab_file:
        with open(vocab_file, 'rb') as f:
            x2i, i2x = pickle.load(f)
    else:
        x2i, i2x = get_index_mappings(sentences)

    # Convert sentences to indices
    tokenized = tokenize_sentences(sentences, x2i)

    if return_sentences:
        return tokenized, sentences, x2i, i2x
    return tokenized, x2i, i2x


def split_train_test(data, split=0.8):
    """
    randomly shuffle data with RANDOM_SEED and split into train and validation set.
    Args:
        data: list of data
        split: proportion of first split.

    Returns: two lists of randomly shuffled data.

    """
    np.random.seed = RANDOM_SEED
    np.random.shuffle(data)
    split_idx = int(len(data) * split)
    return data[:split_idx], data[split_idx:]


def prepare_batch(batch, pad_idx):
    """
    Prepare batch for training

    Args:
        batch: batch of sentences
        pad_idx: idx to pad sentences with, usually 0.

    Returns: 3 torch.autograd.Variable, input words and tags, output arcs.
             Additionally returns array with sentence lengths lengths (for padding).
    """
    # for padding, get lenghts in order big to small,
    # and get max length
    lengths_idx = np.argsort([len(b) for b in batch])[::-1]
    batch = [batch[i] for i in lengths_idx]
    lengths = np.array([len(b) for b in batch], dtype=int)
    max_len = len(batch[0])

    # generate words and tags
    words = np.full([len(batch), max_len], pad_idx, dtype=int)
    tags = np.full([len(batch), max_len], pad_idx, dtype=int)
    for i, sentence in enumerate(batch):
        for j, word in enumerate(sentence):
            words[i, j] = word[1]
            tags[i, j] = word[2]

    words = Variable(torch.from_numpy(words))
    tags = Variable(torch.from_numpy(tags))

    # generate arcs
    # each arc is [batch_idx, head_idx, dep_idx, label_idx]
    arcs = []
    for i, sentence in enumerate(batch):
        # Skip first word, root has no incoming arc.
        for word in sentence[1:]:
            arc = [i, word[3], word[0], word[4]]
            arcs.append(arc)
    arcs = np.vstack(arcs)
    return words, tags, arcs, lengths


def chunks(l, n):
    """
    Split list l in to evenly sized chunks of size n.
    source: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    Args:
        l: list
        n: chunk size

    Yields: chunks of size n
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def batch_loader(data, batch_size, pad_idx=0, shuffle=True, repeat=True):
    """
    Batch generator for arc parser
    Args:
        data: list of parsed and indexed sentences
        batch_size: size of batches to yield
        pad_idx: idx to pad sentences with, usually 0.
        shuffle: if True, shuffle data each iteration.
        repeat: if repeat, allow multiple iterations over data

    Returns:

    """
    idx = np.arange(len(data))
    np.random.seed = RANDOM_SEED
    while True:
        if shuffle:
            np.random.shuffle(idx)
        for chunk in chunks(idx, batch_size):
            batch = [data[i] for i in chunk]
            yield prepare_batch(batch, pad_idx)
        if not repeat:
            break
