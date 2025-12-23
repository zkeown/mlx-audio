"""Tests for Whisper tokenizer."""

import pytest

# Check if tiktoken is available
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


@pytest.mark.skipif(not HAS_TIKTOKEN, reason="tiktoken not installed")
class TestWhisperTokenizer:
    """Tests for WhisperTokenizer."""

    def test_init_multilingual(self):
        """Test multilingual tokenizer initialization."""
        from mlx_audio.models.whisper.tokenizer import WhisperTokenizer

        tokenizer = WhisperTokenizer(multilingual=True)
        assert tokenizer.multilingual is True

    def test_init_english_only(self):
        """Test English-only tokenizer initialization."""
        from mlx_audio.models.whisper.tokenizer import WhisperTokenizer

        tokenizer = WhisperTokenizer(multilingual=False)
        assert tokenizer.multilingual is False

    def test_special_tokens(self):
        """Test special token IDs are set."""
        from mlx_audio.models.whisper.tokenizer import WhisperTokenizer

        tokenizer = WhisperTokenizer(multilingual=True)

        assert tokenizer.eot is not None
        assert tokenizer.sot is not None
        assert tokenizer.transcribe is not None
        assert tokenizer.translate is not None
        assert tokenizer.no_timestamps is not None
        assert tokenizer.no_speech is not None

    def test_encode_decode_roundtrip(self):
        """Test encode-decode preserves text."""
        from mlx_audio.models.whisper.tokenizer import WhisperTokenizer

        tokenizer = WhisperTokenizer(multilingual=True)

        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert decoded == text

    def test_decode_skip_special_tokens(self):
        """Test decode can skip special tokens."""
        from mlx_audio.models.whisper.tokenizer import WhisperTokenizer

        tokenizer = WhisperTokenizer(multilingual=True)

        # Create a sequence with special tokens
        tokens = [tokenizer.sot, tokenizer.transcribe] + tokenizer.encode("test")

        # With skip_special_tokens=True (default)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        assert "test" in decoded.lower()

    def test_language_token(self):
        """Test language token retrieval."""
        from mlx_audio.models.whisper.tokenizer import WhisperTokenizer

        tokenizer = WhisperTokenizer(multilingual=True)

        en_token = tokenizer.get_language_token("en")
        zh_token = tokenizer.get_language_token("zh")

        assert en_token != zh_token
        assert isinstance(en_token, int)

    def test_language_token_by_name(self):
        """Test language token retrieval by full name."""
        from mlx_audio.models.whisper.tokenizer import WhisperTokenizer

        tokenizer = WhisperTokenizer(multilingual=True)

        en_token_code = tokenizer.get_language_token("en")
        en_token_name = tokenizer.get_language_token("english")

        assert en_token_code == en_token_name

    def test_language_token_invalid(self):
        """Test invalid language raises error."""
        from mlx_audio.models.whisper.tokenizer import WhisperTokenizer

        tokenizer = WhisperTokenizer(multilingual=True)

        with pytest.raises(ValueError, match="Unknown language"):
            tokenizer.get_language_token("invalid")

    def test_language_token_english_only_raises(self):
        """Test language token not available for English-only model."""
        from mlx_audio.models.whisper.tokenizer import WhisperTokenizer

        tokenizer = WhisperTokenizer(multilingual=False)

        with pytest.raises(ValueError, match="not available"):
            tokenizer.get_language_token("en")

    def test_all_language_tokens(self):
        """Test all_language_tokens property."""
        from mlx_audio.models.whisper.tokenizer import WhisperTokenizer, LANGUAGES

        tokenizer = WhisperTokenizer(multilingual=True)

        all_tokens = tokenizer.all_language_tokens
        assert len(all_tokens) == len(LANGUAGES)
        assert all(isinstance(t, int) for t in all_tokens)

    def test_get_initial_tokens_transcribe(self):
        """Test initial tokens for transcription."""
        from mlx_audio.models.whisper.tokenizer import WhisperTokenizer

        tokenizer = WhisperTokenizer(multilingual=True)

        tokens = tokenizer.get_initial_tokens(
            language="en",
            task="transcribe",
            timestamps=True,
        )

        assert tokenizer.sot in tokens
        assert tokenizer.transcribe in tokens
        assert tokenizer.no_timestamps not in tokens

    def test_get_initial_tokens_translate(self):
        """Test initial tokens for translation."""
        from mlx_audio.models.whisper.tokenizer import WhisperTokenizer

        tokenizer = WhisperTokenizer(multilingual=True)

        tokens = tokenizer.get_initial_tokens(
            language="zh",
            task="translate",
            timestamps=False,
        )

        assert tokenizer.sot in tokens
        assert tokenizer.translate in tokens
        assert tokenizer.no_timestamps in tokens

    def test_is_timestamp(self):
        """Test timestamp token detection."""
        from mlx_audio.models.whisper.tokenizer import WhisperTokenizer

        tokenizer = WhisperTokenizer(multilingual=True)

        # Timestamp tokens should be in range
        assert tokenizer.is_timestamp(tokenizer.timestamp_begin)
        assert tokenizer.is_timestamp(tokenizer.timestamp_end)

        # Non-timestamp tokens should return False
        assert not tokenizer.is_timestamp(tokenizer.sot)
        assert not tokenizer.is_timestamp(tokenizer.eot)

    def test_timestamp_to_seconds(self):
        """Test timestamp token to seconds conversion."""
        from mlx_audio.models.whisper.tokenizer import WhisperTokenizer

        tokenizer = WhisperTokenizer(multilingual=True)

        # First timestamp should be 0.0
        t0 = tokenizer.timestamp_to_seconds(tokenizer.timestamp_begin)
        assert t0 == 0.0

        # Last timestamp should be 30.0
        t30 = tokenizer.timestamp_to_seconds(tokenizer.timestamp_end)
        assert t30 == 30.0

    def test_seconds_to_timestamp(self):
        """Test seconds to timestamp token conversion."""
        from mlx_audio.models.whisper.tokenizer import WhisperTokenizer

        tokenizer = WhisperTokenizer(multilingual=True)

        # 0 seconds
        t0 = tokenizer.seconds_to_timestamp(0.0)
        assert t0 == tokenizer.timestamp_begin

        # 30 seconds
        t30 = tokenizer.seconds_to_timestamp(30.0)
        assert t30 == tokenizer.timestamp_end

        # 15 seconds (middle)
        t15 = tokenizer.seconds_to_timestamp(15.0)
        assert tokenizer.timestamp_to_seconds(t15) == pytest.approx(15.0, abs=0.02)

    def test_timestamp_roundtrip(self):
        """Test seconds -> token -> seconds roundtrip."""
        from mlx_audio.models.whisper.tokenizer import WhisperTokenizer

        tokenizer = WhisperTokenizer(multilingual=True)

        for seconds in [0.0, 5.5, 15.0, 25.0, 30.0]:
            token = tokenizer.seconds_to_timestamp(seconds)
            recovered = tokenizer.timestamp_to_seconds(token)
            assert recovered == pytest.approx(seconds, abs=0.02)


@pytest.mark.skipif(not HAS_TIKTOKEN, reason="tiktoken not installed")
class TestGetTokenizer:
    """Tests for cached tokenizer retrieval."""

    def test_get_tokenizer_cached(self):
        """Test get_tokenizer returns cached instance."""
        from mlx_audio.models.whisper.tokenizer import get_tokenizer

        t1 = get_tokenizer(multilingual=True, language="en")
        t2 = get_tokenizer(multilingual=True, language="en")

        assert t1 is t2  # Same instance

    def test_get_tokenizer_different_params(self):
        """Test get_tokenizer returns different instances for different params."""
        from mlx_audio.models.whisper.tokenizer import get_tokenizer

        t1 = get_tokenizer(multilingual=True, language="en")
        t2 = get_tokenizer(multilingual=True, language="zh")

        # Different language = potentially different instance
        # (cache uses all params as key)
        assert t1.language != t2.language


class TestLanguages:
    """Tests for language constants."""

    def test_languages_dict(self):
        """Test LANGUAGES dictionary."""
        from mlx_audio.models.whisper.tokenizer import LANGUAGES

        assert "en" in LANGUAGES
        assert LANGUAGES["en"] == "english"
        assert "zh" in LANGUAGES
        assert LANGUAGES["zh"] == "chinese"

    def test_to_language_code(self):
        """Test TO_LANGUAGE_CODE reverse mapping."""
        from mlx_audio.models.whisper.tokenizer import TO_LANGUAGE_CODE

        assert TO_LANGUAGE_CODE["english"] == "en"
        assert TO_LANGUAGE_CODE["chinese"] == "zh"
