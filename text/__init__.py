import numpy as np
import cmudict
from g2p_en import G2p
import re
from .normalize_numbers import normalize_numbers

_punctuation = '!\'()? '
_special_punctuation = ',.:;-'

symbols = cmudict.symbols() + list(_special_punctuation) + list(_punctuation)
# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]

_whitespace_re = re.compile(r'\s+')
f_g2p = G2p()

def _expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def _clean_text(text):
    text = text.lower()
    text = normalize_numbers(text)
    text = re.sub("--", "-", text)
    text = _expand_abbreviations(text)
    return re.sub(_whitespace_re, ' ', text)

def _g2p(text):
    """Convert grapheme to phoneme."""
    phones = f_g2p(text)
    new_phones = []
    for i in range(len(phones)):
        if phones[i] == " ":
            try:
                if phones[i + 1] in _punctuation: continue
            except IndexError: pass
        new_phones.append(phones[i])
    return new_phones

def encode_text(text):
    phones = _g2p(_clean_text(text))
    sequence = [_symbol_to_id[p] for p in phones]
    return np.array(sequence)