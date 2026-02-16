class CharacterTokenizer:
    def __init__(self, text: str) -> None:
        # Extract every unique character to build the "Alphabet"
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # Create look-up tables (Dictionaries)
        # 'a' -> 1, 'b' -> 2...
        self.stoi = { ch:i for i,ch in enumerate(chars) }

        # 1 -> 'a', 2 -> 'b'...
        self.itos = { i:ch for i,ch in enumerate(chars) }

    def encode(self, s: str) -> list[int]:
        # String -> List of Integers
        return [self.stoi[c] for c in s]

    def decode(self, l: list[int]) -> str:
        # List of Integers -> String
        return ''.join([self.itos[i] for i in l])

