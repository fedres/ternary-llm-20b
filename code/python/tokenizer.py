"""
Ternary Model Tokenizer Integration
A high-performance BPE tokenizer with 100k vocabulary optimized for ternary models.
Supports multi-sequence processing, evaluation datasets, and model integration.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor


@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer."""
    vocab_size: int = 100000
    special_tokens: List[str] = None
    byte_fallback: bool = True
    cache_size: int = 10000
    normalization: str = "nfkc"
    min_frequency: int = 2
    max_token_length: int = 100
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]


@dataclass
class BatchEncoding:
    """Batch tokenization result with attention masking."""
    input_ids: np.ndarray
    attention_mask: np.ndarray
    token_type_ids: Optional[np.ndarray] = None
    special_token_mask: Optional[np.ndarray] = None
    
    @property
    def max_length(self) -> int:
        return self.input_ids.shape[1]
    
    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer optimized for ternary models.
    Features:
    - 100k vocabulary size
    - Special token handling
    - Multi-sequence batch processing
    - Performance optimizations
    - Evaluation dataset support
    """
    
    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        self.vocab_size = self.config.vocab_size
        self.special_tokens = self.config.special_tokens
        
        # Create special token mappings
        self.special_token_map = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.vocab = {}
        self.merges = []
        self.unk_token = "[UNK]"
        
        # Caching for performance
        self._cache = {}
        self._max_cache_size = self.config.cache_size
        self._cache_counter = 0
        
        # Model integration helpers
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.mask_token_id = 3
        self.unk_token_id = self.special_token_map[self.unk_token]
        
    def normalize(self, text: str) -> str:
        """Normalize text using specified normalization."""
        if self.config.normalization == "nfkc":
            text = unicodedata.normalize('NFKC', text)
        elif self.config.normalization == "nfc":
            text = unicodedata.normalize('NFC', text)
        
        # Basic whitespace normalization
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def encode_pairs(self, text: str) -> List[str]:
        """Encode text to character pairs for BPE training."""
        text = self.normalize(text)
        
        # Convert to list of characters with word boundaries
        words = text.split()
        word_pairs = []
        
        for word in words:
            # Add word boundary markers
            chars = list(word)
            chars.append('</w>')  # End of word marker
            word_pairs.append(chars)
        
        # Get all character pairs
        pairs = set()
        for word in word_pairs:
            for i in range(len(word) - 1):
                pairs.add((word[i], word[i + 1]))
        
        return list(pairs)
    
    def get_stats(self, word_pairs: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """Calculate statistics for character pairs."""
        pairs = defaultdict(int)
        for word in word_pairs:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += 1
        return pairs
    
    def merge_pair(self, word_pairs: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
        """Merge a character pair in all words."""
        new_word_pairs = []
        for word in word_pairs:
            i = 0
            new_word = []
            while i < len(word):
                if (i < len(word) - 1 and 
                    word[i] == pair[0] and 
                    word[i + 1] == pair[1]):
                    # Merge the pair
                    new_word.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_pairs.append(new_word)
        
        return new_word_pairs
    
    def train_bpe(self, texts: List[str], vocab_size: int = None) -> None:
        """Train BPE tokenizer on given texts."""
        if vocab_size is None:
            vocab_size = self.vocab_size
        
        print(f"Training BPE tokenizer with {vocab_size} vocabulary size...")
        
        # Tokenize all texts into words
        words = []
        for text in texts:
            text = self.normalize(text)
            text_words = text.split()
            for word in text_words:
                chars = list(word)
                chars.append('</w>')
                words.append(chars)
        
        # Get vocabulary from characters (excluding special tokens)
        vocab = Counter()
        for word in words:
            for char in word:
                vocab[char] += 1
        
        # Remove characters with low frequency and convert to list for sorting
        vocab_items = [(k, v) for k, v in vocab.items() if v >= self.config.min_frequency]
        vocab_items.sort(key=lambda x: x[1], reverse=True)
        
        # Create initial vocabulary
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        
        # Add frequent characters
        char_tokens = [token for token, _ in vocab_items[:vocab_size - len(self.special_tokens)]]
        for token in char_tokens:
            self.vocab[token] = len(self.vocab)
        
        # Train BPE merges
        num_merges = vocab_size - len(self.vocab)
        for i in range(num_merges):
            # Get pairs statistics
            pairs = self.get_stats(words)
            
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            
            # Merge the pair
            words = self.merge_pair(words, best_pair)
            
            # Add merged token to vocabulary
            merged_token = best_pair[0] + best_pair[1]
            self.vocab[merged_token] = len(self.vocab)
            
            if (i + 1) % 1000 == 0:
                print(f"Completed {i + 1}/{num_merges} merges...")
        
        print(f"BPE training completed. Vocabulary size: {len(self.vocab)}")
    
    def bpe_encode(self, word: str) -> List[str]:
        """Encode a single word using trained BPE merges."""
        if word in self.vocab:
            return [word]
        
        word = self.normalize(word)
        chars = list(word) + ['</w>']
        
        # Get all possible pairs
        pairs = []
        for i in range(len(chars) - 1):
            pairs.append((chars[i], chars[i + 1]))
        
        if not pairs:
            return [chars[0] if chars else word]
        
        # Apply BPE merges
        while True:
            # Find the best pair to merge (highest frequency in vocab)
            best_pair = None
            best_score = -1
            
            for pair in pairs:
                if pair[0] + pair[1] in self.vocab:
                    score = len(self.vocab) - self.vocab[pair[0] + pair[1]]
                    if score > best_score:
                        best_score = score
                        best_pair = pair
            
            if best_pair is None:
                break
            
            # Merge the best pair
            new_chars = []
            i = 0
            while i < len(chars):
                if (i < len(chars) - 1 and 
                    chars[i] == best_pair[0] and 
                    chars[i + 1] == best_pair[1]):
                    new_chars.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            
            chars = new_chars
            
            # Update pairs
            new_pairs = []
            for i in range(len(chars) - 1):
                new_pairs.append((chars[i], chars[i + 1]))
            pairs = new_pairs
            
            if not pairs:
                break
        
        return chars
    
    @lru_cache(maxsize=1000)
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using BPE encoding."""
        # Check cache first
        cache_key = hash(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        text = self.normalize(text)
        tokens = []
        
        # Split by whitespace to preserve sentence boundaries
        words = text.split()
        
        for word in words:
            # Get BPE encoding for word
            bpe_tokens = self.bpe_encode(word)
            
            for token in bpe_tokens:
                # Remove word boundary marker
                if token.endswith('</w>'):
                    token = token[:-4]
                
                if token:
                    if token in self.vocab:
                        tokens.append(token)
                    else:
                        # Fall back to character-level encoding
                        for char in token:
                            if char in self.vocab:
                                tokens.append(char)
                            else:
                                tokens.append(self.unk_token)
        
        # Update cache with LRU
        self._update_cache(cache_key, tokens)
        return tokens
    
    def encode(self, text: Union[str, List[str]], 
               add_special_tokens: bool = True,
               max_length: Optional[int] = None,
               padding: bool = False,
               truncation: bool = False) -> Union[List[int], np.ndarray]:
        """
        Encode text(s) to token IDs.
        
        Args:
            text: Single text string or list of texts
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            
        Returns:
            Token IDs or batch of token IDs
        """
        if isinstance(text, str):
            return self._encode_single(text, add_special_tokens, max_length, padding, truncation)
        else:
            return self._encode_batch(text, add_special_tokens, max_length, padding, truncation)
    
    def _encode_single(self, text: str, add_special_tokens: bool,
                      max_length: Optional[int], padding: bool, 
                      truncation: bool) -> List[int]:
        """Encode a single text."""
        tokens = self.tokenize(text)
        
        # Convert to token IDs
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        # Add special tokens
        if add_special_tokens:
            token_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
        
        # Truncation
        if truncation and max_length is not None:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
        
        # Padding (for single text, this would be done in batch processing)
        return token_ids
    
    def _encode_batch(self, texts: List[str], add_special_tokens: bool,
                     max_length: Optional[int], padding: bool,
                     truncation: bool) -> np.ndarray:
        """Encode batch of texts with padding."""
        # Tokenize all texts
        all_tokens = []
        all_lengths = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.tokenize, text) for text in texts]
            all_tokens = [future.result() for future in futures]
        
        # Convert to token IDs
        all_token_ids = []
        for tokens in all_tokens:
            token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
            all_token_ids.append(token_ids)
            all_lengths.append(len(token_ids))
        
        # Add special tokens
        if add_special_tokens:
            all_token_ids = [
                [self.cls_token_id] + token_ids + [self.sep_token_id] 
                for token_ids in all_token_ids
            ]
            all_lengths = [length + 2 for length in all_lengths]
        
        # Determine padding length
        if padding:
            if max_length is None:
                max_length = max(all_lengths)
        elif max_length is None:
            max_length = min(max(all_lengths), 512)  # Default max length
        
        # Apply truncation
        if truncation:
            all_token_ids = [
                token_ids[:max_length - 2] + [self.sep_token_id] 
                if len(token_ids) > max_length 
                else token_ids
                for token_ids in all_token_ids
            ]
            all_lengths = [min(length, max_length) for length in all_lengths]
        
        # Pad sequences
        if padding:
            padded_ids = []
            attention_mask = []
            
            for token_ids in all_token_ids:
                if len(token_ids) < max_length:
                    # Pad with pad token
                    padded_token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
                    attention_mask.append([1] * len(token_ids) + [0] * (max_length - len(token_ids)))
                else:
                    padded_token_ids = token_ids
                    attention_mask.append([1] * max_length)
                
                padded_ids.append(padded_token_ids)
            
            return np.array(padded_ids, dtype=np.int32)
        else:
            return np.array(all_token_ids, dtype=np.int32)
    
    def batch_encode_plus(self, texts: List[str], 
                         max_length: Optional[int] = None,
                         padding: str = "max_length",
                         truncation: bool = True,
                         add_special_tokens: bool = True) -> BatchEncoding:
        """
        Batch encode texts with comprehensive information.
        
        Returns:
            BatchEncoding object with input_ids and attention_mask
        """
        # Determine padding strategy
        if padding == "max_length":
            if max_length is None:
                # Calculate max length from texts
                lengths = []
                for text in texts:
                    tokens = self.tokenize(text)
                    length = len(tokens)
                    if add_special_tokens:
                        length += 2
                    lengths.append(length)
                max_length = min(max(lengths), 512)
            padding = True
        elif padding == "do_not_pad":
            padding = False
        elif padding == "longest":
            padding = False  # Handle longest padding in encoding
        else:
            raise ValueError(f"Invalid padding strategy: {padding}")
        
        # Encode texts
        input_ids = self.encode(
            texts, 
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            padding=padding,
            truncation=truncation
        )
        
        # Create attention mask
        if padding:
            attention_mask = (input_ids != self.pad_token_id).astype(np.int32)
        else:
            # For variable length sequences
            attention_mask = np.ones_like(input_ids, dtype=np.int32)
        
        return BatchEncoding(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    def decode(self, token_ids: Union[List[int], np.ndarray], 
               skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        tokens = []
        for token_id in token_ids:
            if token_id < len(self.vocab):
                # Get token from reverse vocab
                token = self._reverse_vocab[token_id]
                
                if skip_special_tokens and token in self.special_tokens:
                    continue
                
                # Remove word boundary marker
                if token.endswith('</w>'):
                    token = token[:-4]
                
                tokens.append(token)
        
        # Join tokens and clean up
        text = ' '.join(tokens)
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def prepare_for_model(self, texts: List[str], max_length: int = 512) -> Dict[str, np.ndarray]:
        """
        Prepare inputs for ternary model training/inference.
        
        Returns:
            Dictionary with input_ids, attention_mask ready for model
        """
        batch_encoding = self.batch_encode_plus(
            texts=texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True
        )
        
        return {
            "input_ids": batch_encoding.input_ids,
            "attention_mask": batch_encoding.attention_mask,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.sep_token_id,
        }
    
    def prepare_kv_cache(self, input_ids: np.ndarray) -> Dict[str, np.ndarray]:
        """Prepare key-value cache tensors for model inference."""
        batch_size, seq_length = input_ids.shape
        
        return {
            "past_keys": np.zeros((batch_size, 0, 768), dtype=np.float32),
            "past_values": np.zeros((batch_size, 0, 768), dtype=np.float32),
            "attention_mask": np.ones((batch_size, seq_length), dtype=np.int32),
        }
    
    def calculate_perplexity(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculate perplexity for evaluation.
        
        Args:
            predictions: Model predictions (logits)
            targets: Target token IDs
            
        Returns:
            Perplexity score
        """
        # Cross-entropy loss
        cross_entropy = self._cross_entropy_loss(predictions, targets)
        
        # Perplexity = exp(cross_entropy)
        perplexity = np.exp(np.mean(cross_entropy))
        return float(perplexity)
    
    def _cross_entropy_loss(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute cross-entropy loss."""
        # Apply softmax to get probabilities
        exp_logits = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Get log probabilities for target tokens
        log_probs = np.log(probabilities + 1e-8)  # Add epsilon for numerical stability
        
        # Gather log probabilities for target tokens
        target_log_probs = np.take_along_axis(
            log_probs, 
            targets[:, :, np.newaxis], 
            axis=-1
        ).squeeze(-1)
        
        # Mask out padding tokens
        mask = (targets != self.pad_token_id).astype(np.float32)
        masked_log_probs = target_log_probs * mask
        
        # Average over non-masked tokens
        return -masked_log_probs.sum(axis=-1) / (mask.sum(axis=-1) + 1e-8)
    
    def create_evaluation_prompts(self, dataset: str, task_type: str = "zero_shot") -> List[Dict[str, str]]:
        """Create evaluation prompts for different datasets and tasks."""
        prompts = []
        
        if dataset.lower() == "wikitext-2":
            if task_type == "perplexity":
                prompts = [
                    {"text": text, "task": "perplexity"}
                    for text in self._get_wikitext_samples(100)
                ]
        
        elif dataset.lower() == "c4":
            if task_type == "zero_shot":
                prompts = [
                    {"text": f"Question: {text}\nAnswer:", "task": "zero_shot_qa"}
                    for text in self._get_c4_samples(50)
                ]
        
        elif dataset.lower() == "lambada":
            if task_type == "completion":
                prompts = [
                    {"text": self._format_lambada_prompt(text), "task": "completion"}
                    for text in self._get_lambada_samples(100)
                ]
        
        return prompts
    
    def _get_wikitext_samples(self, num_samples: int) -> List[str]:
        """Get sample texts from WikiText-2 for evaluation."""
        # This would typically load from actual dataset
        sample_texts = [
            "The history of the English language is complex and varied.",
            "Natural language processing has advanced significantly in recent years.",
            "Machine learning models require large amounts of training data.",
            "Transformer architectures have revolutionized the field of AI.",
            "Text generation requires understanding of context and semantics.",
        ]
        
        # Repeat and vary samples to reach desired count
        texts = []
        for i in range(num_samples):
            base_text = sample_texts[i % len(sample_texts)]
            # Add some variation
            if i % 3 == 0:
                texts.append(f"According to recent studies, {base_text.lower()}")
            elif i % 3 == 1:
                texts.append(f"Research shows that {base_text}")
            else:
                texts.append(f"{base_text} This finding has important implications.")
        
        return texts
    
    def _get_c4_samples(self, num_samples: int) -> List[str]:
        """Get sample texts from C4 for evaluation."""
        sample_questions = [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "What causes climate change?",
            "Explain the concept of gravity.",
            "What is machine learning?",
            "How does the human brain work?",
            "What is quantum physics?",
            "Explain the theory of evolution.",
        ]
        
        return sample_questions * (num_samples // len(sample_questions) + 1)
    
    def _get_lambada_samples(self, num_samples: int) -> List[str]:
        """Get sample texts from LAMBADA for evaluation."""
        sample_completions = [
            "The chef prepared a delicious meal that everyone",
            "She opened the door and saw her friend waiting",
            "The scientist conducted experiments in the",
            "The children played games in the",
            "The teacher explained the lesson to the",
        ]
        
        return sample_completions * (num_samples // len(sample_completions) + 1)
    
    def _format_lambada_prompt(self, text: str) -> str:
        """Format LAMBADA text for completion task."""
        # Split at the last space to create a gap for prediction
        words = text.split()
        if len(words) > 1:
            prompt = " ".join(words[:-1])
            return prompt
        return text
    
    def _update_cache(self, key: str, value: List[str]) -> None:
        """Update LRU cache."""
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = min(self._cache.keys())
            del self._cache[oldest_key]
        
        self._cache[key] = value
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save tokenizer to directory."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save vocabulary and merges
        vocab_file = os.path.join(save_directory, "vocab.json")
        merges_file = os.path.join(save_directory, "merges.txt")
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        
        # Save vocab
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # Save merges
        with open(merges_file, 'w', encoding='utf-8') as f:
            for merge in self.merges:
                f.write(f"{merge[0]} {merge[1]}\n")
        
        # Save config
        config_dict = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "pad_token_id": self.pad_token_id,
            "cls_token_id": self.cls_token_id,
            "sep_token_id": self.sep_token_id,
            "mask_token_id": self.mask_token_id,
            "unk_token_id": self.unk_token_id,
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
    
    def from_pretrained(self, save_directory: str) -> 'BPETokenizer':
        """Load tokenizer from directory."""
        vocab_file = os.path.join(save_directory, "vocab.json")
        merges_file = os.path.join(save_directory, "merges.txt")
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        
        # Load vocab
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        # Load merges
        self.merges = []
        if os.path.exists(merges_file):
            with open(merges_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 2:
                            self.merges.append((parts[0], parts[1]))
        
        # Load config
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.pad_token_id = config.get("pad_token_id", 0)
                self.cls_token_id = config.get("cls_token_id", 1)
                self.sep_token_id = config.get("sep_token_id", 2)
                self.mask_token_id = config.get("mask_token_id", 3)
                self.unk_token_id = config.get("unk_token_id", 4)
        
        # Create reverse vocab
        self._reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        return self
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    @property
    def vocab_list(self) -> List[str]:
        """Return list of vocabulary tokens."""
        return [token for token, _ in sorted(self.vocab.items(), key=lambda x: x[1])]


# Convenience functions for easy usage
def create_tokenizer(config: TokenizerConfig = None) -> BPETokenizer:
    """Create a new BPE tokenizer."""
    return BPETokenizer(config)


def train_tokenizer(texts: List[str], vocab_size: int = 100000, 
                   special_tokens: List[str] = None) -> BPETokenizer:
    """Train a BPE tokenizer on given texts."""
    config = TokenizerConfig(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    tokenizer = BPETokenizer(config)
    tokenizer.train_bpe(texts, vocab_size)
    
    # Create reverse vocab
    tokenizer._reverse_vocab = {idx: token for token, idx in tokenizer.vocab.items()}
    
    return tokenizer


# Example usage
if __name__ == "__main__":
    # Example training data
    training_texts = [
        "This is a sample text for training the tokenizer.",
        "Language models require large amounts of training data.",
        "Byte-pair encoding is a popular tokenization method.",
        "Natural language processing has advanced significantly.",
        "Machine learning models can generate human-like text.",
    ]
    
    # Create and train tokenizer
    tokenizer = train_tokenizer(
        texts=training_texts,
        vocab_size=10000,  # Smaller for example
        special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
    )
    
    # Test tokenization
    text = "Hello, world! This is a test of the tokenizer."
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    
    print(f"Original text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {tokenizer.decode(token_ids)}")
    
    # Test batch processing
    texts = [
        "First example text for batch processing.",
        "Second example with different content.",
        "Third example to test the tokenizer."
    ]
    
    batch_encoding = tokenizer.batch_encode_plus(texts, max_length=50, padding="max_length")
    print(f"\nBatch input shape: {batch_encoding.input_ids.shape}")
    print(f"Batch attention mask shape: {batch_encoding.attention_mask.shape}")
    
    # Test model preparation
    model_inputs = tokenizer.prepare_for_model(texts, max_length=100)
    print(f"\nModel input keys: {list(model_inputs.keys())}")
    print(f"Input IDs shape: {model_inputs['input_ids'].shape}")
    
    # Save tokenizer
    tokenizer.save_pretrained("./tokenizer_save")
    print("\nTokenizer saved successfully!")