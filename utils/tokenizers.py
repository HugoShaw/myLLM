import re
import os
import json
import regex
import requests
from tqdm import tqdm
from functools import lru_cache

class SimpleTokenizerV1:
    """this version doesnt consider unknown tokens, so it is really simple and will only be used in learning
    """
    def __init__(self, vocab):
        # Initialize the tokenizer with a vocabulary
        # vocab is a dictionary mapping strings to integers
        self.str_to_int = vocab
        # Create a reverse mapping from integers to strings
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text):
        # Preprocess the text by splitting on specified punctuation and whitespace
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        # Strip any extra whitespace and filter out empty strings
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        # Convert each preprocessed token to its corresponding integer ID
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        # Convert each integer ID back to its corresponding string token
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before specified punctuation marks
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

class SimpleTokenizerV2:
    """consider unknown tokens but still simple
    """
    def __init__(self, vocab):
        # Initialize the tokenizer with a vocabulary
        # vocab is a dictionary mapping strings to integers
        self.str_to_int = vocab
        # Create a reverse mapping from integers to strings
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text):
        # Preprocess the text by splitting on specified punctuation and whitespace
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        # Strip any extra whitespace and filter out empty strings
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Check if each preprocessed token is in the vocabulary, else replace it with an empty string
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]
        # Convert each valid preprocessed token to its corresponding integer ID
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        # Convert each integer ID back to its corresponding string token
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before specified punctuation marks
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


######################## build GPT2 - BPE from scratch ##############################
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    # Create lists of byte values for certain unicode ranges
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    # Convert byte values to corresponding unicode characters
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        # Initialize the Encoder with the provided encoder dictionary and BPE merges
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}  # Reverse mapping for decoding
        self.errors = errors  # How to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()  # Byte-to-unicode lookup table
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}  # Unicode-to-byte lookup table
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))  # BPE merges with their ranks
        self.cache = {}  # Cache for BPE tokenization
        
        # Regex pattern for tokenization
        self.pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    def bpe(self, token):
        # Apply Byte Pair Encoding (BPE) to the given token
        if token in self.cache:
            return self.cache[token]
        
        word = tuple(token)
        pairs = get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        # Encode the given text using BPE
        bpe_tokens = []
        for token in regex.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        # Decode the given list of BPE tokens
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

def get_encoder(model_name, models_dir):
    # Load the encoder and BPE merges from the specified model directory
    with open(os.path.join(models_dir, model_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(models_dir, model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(encoder=encoder, bpe_merges=bpe_merges)


def download_vocab():
    # Modified code from
    subdir = r'./gpt2_model'
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace('\\', '/')  # needed for Windows

    for filename in ['encoder.json', 'vocab.bpe']:
        r = requests.get("https://openaipublic.blob.core.windows.net/gpt-2/models/117M/" + filename, stream=True)

        with open(os.path.join(subdir, filename), 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)



if __name__ == "__main__":
    download_vocab()






