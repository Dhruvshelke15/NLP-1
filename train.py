import collections
import math

class Corpus:
    """
    Handles data loading, preprocessing, and unknown word replacement.
    This class prepares the corpus by creating a vocabulary and converting 
    out-of-vocabulary words to <UNK> tokens.
    """
    def __init__(self, train_path, val_path, unk_threshold=1):
        """
        Initializes the Corpus handler.

        Args:
            train_path (str): Path to the training data file.
            val_path (str): Path to the validation data file.
            unk_threshold (int): Words appearing this many times or fewer will be 
                                 replaced by '<UNK>'.
        """
        self.train_path = train_path
        self.val_path = val_path
        self.unk_threshold = unk_threshold
        
        self.vocab = set()
        self.train_corpus = []
        self.val_corpus = []
        
        self._prepare_data()

    def _prepare_data(self):
        """Private method to orchestrate the data loading and preprocessing."""
        print("1. Loading and preparing data...")
        
        # Step 1: Read raw sentences from files
        raw_train_sentences = self._read_file(self.train_path)
        raw_val_sentences = self._read_file(self.val_path)

        # Step 2: Build vocabulary from training data and replace rare words with <UNK>
        self.vocab, self.train_corpus = self._build_vocab_and_unk(raw_train_sentences)
        print(f"   - Vocabulary created. Size (V): {len(self.vocab)}")

        # Step 3: Apply the same vocabulary to the validation set
        self.val_corpus = self._apply_unk_to_corpus(raw_val_sentences, self.vocab)
        print("   - Data preparation complete.\n")

    def _read_file(self, file_path):
        """Reads a file and returns a list of tokenized sentences."""
        with open(file_path, 'r', encoding='utf-8') as f:
            # Each line is a review/sentence. Lowercase, strip whitespace, and split into tokens.
            return [line.lower().strip().split() for line in f]

    def _build_vocab_and_unk(self, sentences):
        """Builds vocabulary and replaces rare words with <UNK>."""
        word_counts = collections.Counter(word for sentence in sentences for word in sentence)
        
        # Create a vocabulary of words that appear more than the threshold
        vocab = {word for word, count in word_counts.items() if count > self.unk_threshold}
        
        # Add special tokens to the vocabulary
        vocab.update(['<s>', '</s>', '<UNK>'])
        
        # Replace rare words and add sentence boundary markers
        processed_corpus = []
        for sentence in sentences:
            unk_sentence = [word if word in vocab else '<UNK>' for word in sentence]
            processed_corpus.append(['<s>'] + unk_sentence + ['</s>'])
            
        return vocab, processed_corpus
    
    def _apply_unk_to_corpus(self, sentences, vocab):
        """Applies an existing vocabulary to a new corpus, replacing OOV words with <UNK>."""
        processed_corpus = []
        for sentence in sentences:
            unk_sentence = [word if word in vocab else '<UNK>' for word in sentence]
            processed_corpus.append(['<s>'] + unk_sentence + ['</s>'])
        return processed_corpus


class NgramLanguageModel:
    """
    A class for creating, training, and evaluating n-gram language models.
    """
    def __init__(self, corpus, n_val):
        """
        Initializes and trains the language model.
        
        Args:
            corpus (list of list of str): The preprocessed training corpus.
            n_val (int): The 'n' in n-gram (e.g., 1 for unigram, 2 for bigram).
        """
        self.n = n_val
        self.unigram_counts = collections.Counter()
        self.bigram_counts = collections.Counter()
        self.total_tokens = 0
        self.vocab_size = 0
        self.train(corpus)

    def train(self, corpus):
        """Trains the model by counting n-grams from the corpus."""
        for sentence in corpus:
            self.total_tokens += len(sentence)
            # Unigram counts are needed for both models
            for word in sentence:
                self.unigram_counts[word] += 1
            
            # Bigram counts if it's a bigram model
            if self.n >= 2:
                for i in range(len(sentence) - 1):
                    self.bigram_counts[(sentence[i], sentence[i+1])] += 1
        
        self.vocab_size = len(self.unigram_counts)

    def get_prob(self, tokens, k=0):
        """
        Generic probability function for unigrams or bigrams.
        
        Args:
            tokens (tuple): A tuple representing the n-gram. E.g., ('word',) or ('prev_word', 'word').
            k (float): The Add-k smoothing factor.
            
        Returns:
            float: The calculated probability.
        """
        if len(tokens) == 1:  # Unigram
            word = tokens[0]
            numerator = self.unigram_counts.get(word, 0) + k
            denominator = self.total_tokens + (k * self.vocab_size)
            return numerator / denominator
        
        elif len(tokens) == 2:  # Bigram
            prev_word, word = tokens
            numerator = self.bigram_counts.get((prev_word, word), 0) + k
            denominator = self.unigram_counts.get(prev_word, 0) + (k * self.vocab_size)
            
            if denominator == 0:
                # This can happen if prev_word is unseen and k=0.
                return 0
            return numerator / denominator
        
        else:
            raise ValueError("This model only supports n=1 (unigram) and n=2 (bigram).")

    def perplexity(self, corpus, k=0):
        """Calculates the perplexity of the model on a given corpus."""
        total_log_prob = 0
        num_tokens = 0
        
        for sentence in corpus:
            # We don't count the start token '<s>' in the total token count for perplexity
            num_tokens += len(sentence) - 1
            
            for i in range(1, len(sentence)):
                if self.n == 1:
                    ngram = (sentence[i],)
                elif self.n == 2:
                    ngram = (sentence[i-1], sentence[i])
                else:
                    raise ValueError("This model only supports n=1 and n=2.")
                
                prob = self.get_prob(ngram, k=k)
                
                if prob == 0:
                    total_log_prob += float('-inf')
                    break # A single 0 probability makes the whole sentence probability 0
                else:
                    total_log_prob += math.log2(prob)
            
            if total_log_prob == float('-inf'):
                return float('inf')

        avg_log_prob = total_log_prob / num_tokens
        perplexity = 2 ** (-avg_log_prob)
        return perplexity


# --- Main Execution Block ---

if __name__ == "__main__":
    # --- Setup and Data Preparation ---
    # Per the assignment, you should place train.txt and val.txt in the same directory as this script.
    train_file = 'train.txt'
    val_file = 'val.txt'
    
    # UNK_THRESHOLD=1 means words seen only once will become <UNK>. This is a min frequency of 2.
    corpus_handler = Corpus(train_file, val_file, unk_threshold=1)
    
    # --- Model Training ---
    print("2. Training Language Models...")
    unigram_model = NgramLanguageModel(corpus_handler.train_corpus, n_val=1)
    bigram_model = NgramLanguageModel(corpus_handler.train_corpus, n_val=2)
    print("   - Training complete.\n")

    # --- Evaluation ---
    print("3. Evaluating Models...")
    smoothing_params = [1.0, 0.1, 0.01, 0.001, 0.0] # k values for smoothing, including 0 for unsmoothed

    # Evaluate on Training Set
    print("\n--- Perplexity on TRAINING Set ---")
    print("This helps check for underfitting/overfitting.")
    
    print("\nUnigram Model (Training Set):")
    for k in smoothing_params:
        ppl = unigram_model.perplexity(corpus_handler.train_corpus, k=k)
        label = f"k={k:<5}" if k != 0 else "Unsmoothed"
        print(f"   - {label}: {ppl:.4f}")

    print("\nBigram Model (Training Set):")
    for k in smoothing_params:
        ppl = bigram_model.perplexity(corpus_handler.train_corpus, k=k)
        label = f"k={k:<5}" if k != 0 else "Unsmoothed"
        print(f"   - {label}: {ppl:.4f}")

    # Evaluate on Validation Set
    print("\n--- Perplexity on VALIDATION Set ---")
    print("This is the primary measure of how well the model generalizes.")
    
    print("\nUnigram Model (Validation Set):")
    for k in smoothing_params:
        ppl = unigram_model.perplexity(corpus_handler.val_corpus, k=k)
        label = f"k={k:<5}" if k != 0 else "Unsmoothed"
        print(f"   - {label}: {ppl:.4f}")

    print("\nBigram Model (Validation Set):")
    for k in smoothing_params:
        ppl = bigram_model.perplexity(corpus_handler.val_corpus, k=k)
        label = f"k={k:<5}" if k != 0 else "Unsmoothed"
        print(f"   - {label}: {ppl:.4f}")