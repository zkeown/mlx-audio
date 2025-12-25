"""RoBERTa-based text encoder for CLAP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from mlx_audio.models.clap.config import CLAPTextConfig


class RobertaEmbeddings(nn.Module):
    """RoBERTa embeddings: token + position + token_type.

    Args:
        config: Text encoder configuration
    """

    def __init__(self, config: CLAPTextConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        # Use LayerNorm (capital N) to match HuggingFace naming
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Padding idx for position embeddings
        self.padding_idx = config.pad_token_id

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            input_ids: Token IDs [B, L]
            token_type_ids: Token type IDs [B, L] (optional)
            position_ids: Position IDs [B, L] (optional)

        Returns:
            Embeddings [B, L, hidden_size]
        """
        B, L = input_ids.shape

        if position_ids is None:
            # Create position IDs starting from padding_idx + 1
            position_ids = mx.arange(self.padding_idx + 1, L + self.padding_idx + 1)
            position_ids = mx.broadcast_to(position_ids, (B, L))

        if token_type_ids is None:
            token_type_ids = mx.zeros((B, L), dtype=mx.int32)

        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class RobertaSelfAttention(nn.Module):
    """RoBERTa self-attention layer.

    Args:
        config: Text encoder configuration
    """

    def __init__(self, config: CLAPTextConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            hidden_states: Input [B, L, hidden_size]
            attention_mask: Attention mask [B, 1, 1, L] or [B, L]

        Returns:
            Output [B, L, hidden_size]
        """
        B, L, _ = hidden_states.shape

        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        # Reshape for multi-head attention
        q = mx.reshape(q, (B, L, self.num_heads, self.head_dim))
        k = mx.reshape(k, (B, L, self.num_heads, self.head_dim))
        v = mx.reshape(v, (B, L, self.num_heads, self.head_dim))

        # Transpose to [B, num_heads, L, head_dim]
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        # Attention scores
        attn = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale

        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                # Expand [B, L] to [B, 1, 1, L]
                attention_mask = attention_mask[:, None, None, :]
            # Convert mask to additive (0 for keep, -inf for mask)
            attention_mask = (1.0 - attention_mask) * -1e9
            attn = attn + attention_mask

        attn = mx.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # [B, num_heads, L, head_dim]
        out = mx.transpose(out, (0, 2, 1, 3))  # [B, L, num_heads, head_dim]
        out = mx.reshape(out, (B, L, -1))  # [B, L, hidden_size]

        return out


class RobertaAttention(nn.Module):
    """RoBERTa attention layer with output projection.

    Args:
        config: Text encoder configuration
    """

    def __init__(self, config: CLAPTextConfig):
        super().__init__()
        self.self_attn = RobertaSelfAttention(config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Forward pass with residual connection."""
        attn_output = self.self_attn(hidden_states, attention_mask)
        attn_output = self.output(attn_output)
        attn_output = self.dropout(attn_output)
        hidden_states = self.layer_norm(hidden_states + attn_output)
        return hidden_states


class RobertaIntermediate(nn.Module):
    """RoBERTa intermediate (FFN first layer).

    Args:
        config: Text encoder configuration
    """

    def __init__(self, config: CLAPTextConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.act(self.dense(hidden_states))


class RobertaOutput(nn.Module):
    """RoBERTa output (FFN second layer with residual).

    Args:
        config: Text encoder configuration
    """

    def __init__(self, config: CLAPTextConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class RobertaLayer(nn.Module):
    """Single RoBERTa transformer layer.

    Args:
        config: Text encoder configuration
    """

    def __init__(self, config: CLAPTextConfig):
        super().__init__()
        self.attention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Forward pass."""
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class RobertaEncoder(nn.Module):
    """RoBERTa encoder (stack of transformer layers).

    Args:
        config: Text encoder configuration
    """

    def __init__(self, config: CLAPTextConfig):
        super().__init__()
        self.layers = [RobertaLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Forward pass through all layers."""
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class RobertaPooler(nn.Module):
    """Pooler for RoBERTa that matches HuggingFace naming.

    Args:
        hidden_size: Hidden dimension
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # Take [CLS] token (first token)
        first_token = hidden_states[:, 0, :]
        pooled = self.dense(first_token)
        pooled = mx.tanh(pooled)
        return pooled


class CLAPTextProjection(nn.Module):
    """2-layer MLP projection head for text encoder.

    Args:
        in_dim: Input dimension
        out_dim: Output projection dimension
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=True)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)
        return x


class CLAPTextEncoder(nn.Module):
    """CLAP text encoder based on RoBERTa.

    Encodes text into fixed-size embeddings using a pre-trained
    RoBERTa model with a projection head.

    Args:
        config: Text encoder configuration
        projection_dim: Output projection dimension
    """

    def __init__(self, config: CLAPTextConfig, projection_dim: int = 512):
        super().__init__()
        self.config = config

        # RoBERTa components
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        # Pooler (uses [CLS] token) - wrapped in container for HF naming (pooler.dense)
        self.pooler = RobertaPooler(config.hidden_size)

        # Projection to shared space (2-layer MLP like HuggingFace)
        self.projection = CLAPTextProjection(config.hidden_size, projection_dim)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        token_type_ids: mx.array | None = None,
        normalize: bool = True,
    ) -> mx.array:
        """Forward pass.

        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L] (1 for real, 0 for padding)
            token_type_ids: Token type IDs [B, L]
            normalize: Whether to L2-normalize output

        Returns:
            Text embeddings [B, projection_dim]
        """
        # Embeddings
        hidden_states = self.embeddings(
            input_ids,
            token_type_ids=token_type_ids,
        )

        # Encoder
        hidden_states = self.encoder(hidden_states, attention_mask)

        # Pool using [CLS] token (pooler handles extraction and tanh)
        pooled = self.pooler(hidden_states)

        # Project to shared space
        embeddings = self.projection(pooled)

        if normalize:
            embeddings = embeddings / mx.linalg.norm(embeddings, axis=-1, keepdims=True)

        return embeddings

    def get_embeddings(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Get raw hidden states without pooling.

        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]

        Returns:
            Hidden states [B, L, hidden_size]
        """
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.encoder(hidden_states, attention_mask)
        return hidden_states
