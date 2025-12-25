"""Whisper tokenizer implementation.

Uses tiktoken for BPE encoding, with special token handling for
multilingual transcription and translation tasks.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Language codes supported by multilingual Whisper
LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

# Reverse mapping
TO_LANGUAGE_CODE = {v: k for k, v in LANGUAGES.items()}


class WhisperTokenizer:
    """Whisper BPE tokenizer with special token handling.

    This tokenizer handles:
    - Standard BPE text encoding/decoding via tiktoken
    - Special tokens for task specification (transcribe/translate)
    - Language tokens for multilingual models
    - Timestamp tokens for word-level alignment
    - No-speech detection tokens

    Attributes:
        encoding: The tiktoken encoding instance
        multilingual: Whether this is a multilingual tokenizer
        language: Default language for encoding
        task: Default task (transcribe or translate)
    """

    def __init__(
        self,
        multilingual: bool = True,
        language: str | None = None,
        task: str = "transcribe",
    ):
        """Initialize the tokenizer.

        Args:
            multilingual: Whether to use multilingual vocabulary
            language: Default language code (e.g., "en", "zh")
            task: Default task ("transcribe" or "translate")
        """
        self.multilingual = multilingual
        self.language = language or ("en" if not multilingual else None)
        self.task = task

        # Load tiktoken encoding
        self._encoding = self._get_encoding()

        # Special token IDs
        self._special_tokens = self._init_special_tokens()

    def _get_encoding(self):
        """Get the tiktoken encoding for Whisper."""
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for Whisper tokenizer. "
                "Install with: pip install tiktoken"
            )

        # Whisper uses a custom encoding based on GPT-2 with additions
        if self.multilingual:
            return tiktoken.get_encoding("cl100k_base")
        else:
            return tiktoken.get_encoding("gpt2")

    def _init_special_tokens(self) -> dict[str, int]:
        """Initialize special token mappings.

        Special tokens are appended to the vocabulary in order:
        - <|endoftext|>: End of text
        - <|startoftranscript|>: Start of transcription
        - Language tokens (for multilingual models)
        - <|translate|>: Translation task
        - <|transcribe|>: Transcription task
        - <|startoflm|>: Start of language model
        - <|startofprev|>: Start of previous context
        - <|nospeech|>: No speech detected
        - <|notimestamps|>: Disable timestamps
        - Timestamp tokens <|0.00|> through <|30.00|>
        """
        specials = {}

        # Base vocabulary size depends on multilingual
        if self.multilingual:
            # cl100k_base has ~100k tokens, but Whisper uses custom offsets
            base = 50257  # Standard GPT-2 vocab size as base
        else:
            base = 50257  # GPT-2 vocab size

        idx = base

        # End of text (same position as GPT-2)
        specials["<|endoftext|>"] = 50256

        # Start of transcript
        specials["<|startoftranscript|>"] = idx
        idx += 1

        # Language tokens (for multilingual)
        if self.multilingual:
            for lang_code in LANGUAGES:
                specials[f"<|{lang_code}|>"] = idx
                idx += 1

        # Task tokens
        specials["<|translate|>"] = idx
        idx += 1
        specials["<|transcribe|>"] = idx
        idx += 1

        # Additional special tokens
        specials["<|startoflm|>"] = idx
        idx += 1
        specials["<|startofprev|>"] = idx
        idx += 1
        specials["<|nospeech|>"] = idx
        idx += 1
        specials["<|notimestamps|>"] = idx
        idx += 1

        # Timestamp tokens: <|0.00|> through <|30.00|> in 0.02s increments
        # Total: 1501 timestamp tokens (0.00 to 30.00)
        for i in range(1501):
            t = i * 0.02
            specials[f"<|{t:.2f}|>"] = idx
            idx += 1

        return specials

    @property
    def eot(self) -> int:
        """End of text token ID."""
        return self._special_tokens["<|endoftext|>"]

    @property
    def sot(self) -> int:
        """Start of transcript token ID."""
        return self._special_tokens["<|startoftranscript|>"]

    @property
    def translate(self) -> int:
        """Translation task token ID."""
        return self._special_tokens["<|translate|>"]

    @property
    def transcribe(self) -> int:
        """Transcription task token ID."""
        return self._special_tokens["<|transcribe|>"]

    @property
    def no_timestamps(self) -> int:
        """No timestamps token ID."""
        return self._special_tokens["<|notimestamps|>"]

    @property
    def no_speech(self) -> int:
        """No speech token ID."""
        return self._special_tokens["<|nospeech|>"]

    @property
    def timestamp_begin(self) -> int:
        """First timestamp token ID (<|0.00|>)."""
        return self._special_tokens["<|0.00|>"]

    @property
    def timestamp_end(self) -> int:
        """Last timestamp token ID (<|30.00|>)."""
        return self._special_tokens["<|30.00|>"]

    def get_language_token(self, language: str) -> int:
        """Get token ID for a language.

        Args:
            language: Language code (e.g., "en") or name (e.g., "english")

        Returns:
            Token ID for the language

        Raises:
            ValueError: If language is not supported
        """
        if not self.multilingual:
            raise ValueError("Language tokens not available for English-only models")

        # Convert language name to code if needed
        if language in TO_LANGUAGE_CODE:
            language = TO_LANGUAGE_CODE[language]

        if language not in LANGUAGES:
            raise ValueError(
                f"Unknown language: {language!r}. "
                f"Supported: {list(LANGUAGES.keys())}"
            )

        return self._special_tokens[f"<|{language}|>"]

    @property
    def all_language_tokens(self) -> list[int]:
        """List of all language token IDs."""
        if not self.multilingual:
            return []
        return [self._special_tokens[f"<|{lang}|>"] for lang in LANGUAGES]

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        return self._encoding.encode(text, allowed_special="all")

    def decode(
        self,
        tokens: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text
        """
        if skip_special_tokens:
            # Filter out special tokens
            special_ids = set(self._special_tokens.values())
            tokens = [t for t in tokens if t not in special_ids]

        return self._encoding.decode(tokens)

    def get_initial_tokens(
        self,
        language: str | None = None,
        task: str | None = None,
        timestamps: bool = True,
    ) -> list[int]:
        """Get initial token sequence for decoding.

        Args:
            language: Language code (None for auto-detect)
            task: Task type ("transcribe" or "translate")
            timestamps: Whether to include timestamps

        Returns:
            List of initial token IDs
        """
        task = task or self.task
        tokens = [self.sot]

        # Add language token for multilingual
        if self.multilingual and language is not None:
            tokens.append(self.get_language_token(language))
            # If None, language will be detected during decoding

        # Add task token
        if task == "translate":
            tokens.append(self.translate)
        else:
            tokens.append(self.transcribe)

        # Add no-timestamps token if disabled
        if not timestamps:
            tokens.append(self.no_timestamps)

        return tokens

    def is_timestamp(self, token: int) -> bool:
        """Check if a token is a timestamp token."""
        return self.timestamp_begin <= token <= self.timestamp_end

    def timestamp_to_seconds(self, token: int) -> float:
        """Convert a timestamp token to seconds.

        Args:
            token: Timestamp token ID

        Returns:
            Time in seconds

        Raises:
            ValueError: If token is not a timestamp
        """
        if not self.is_timestamp(token):
            raise ValueError(f"Token {token} is not a timestamp token")
        return (token - self.timestamp_begin) * 0.02

    def seconds_to_timestamp(self, seconds: float) -> int:
        """Convert seconds to timestamp token.

        Args:
            seconds: Time in seconds (0.0 to 30.0)

        Returns:
            Timestamp token ID
        """
        seconds = max(0.0, min(30.0, seconds))
        idx = round(seconds / 0.02)
        return self.timestamp_begin + idx

    def split_to_word_tokens(
        self,
        tokens: list[int],
    ) -> list[tuple[str, list[int]]]:
        """Split tokens into word-token pairs.

        Useful for word-level timestamps.

        Args:
            tokens: List of token IDs

        Returns:
            List of (word, token_ids) pairs
        """
        words = []
        current_word = ""
        current_tokens = []

        for token in tokens:
            # Skip special tokens
            if token in self._special_tokens.values():
                if current_word:
                    words.append((current_word, current_tokens))
                    current_word = ""
                    current_tokens = []
                continue

            text = self._encoding.decode([token])
            current_tokens.append(token)

            # Check if this starts a new word (space prefix)
            if text.startswith(" ") and current_word:
                words.append((current_word, current_tokens[:-1]))
                current_word = text.lstrip()
                current_tokens = [token]
            else:
                current_word += text

        if current_word:
            words.append((current_word, current_tokens))

        return words


@lru_cache(maxsize=4)
def get_tokenizer(
    multilingual: bool = True,
    language: str | None = None,
    task: str = "transcribe",
) -> WhisperTokenizer:
    """Get a cached tokenizer instance.

    Args:
        multilingual: Whether to use multilingual vocabulary
        language: Default language code
        task: Default task

    Returns:
        Cached WhisperTokenizer instance
    """
    return WhisperTokenizer(
        multilingual=multilingual,
        language=language,
        task=task,
    )
