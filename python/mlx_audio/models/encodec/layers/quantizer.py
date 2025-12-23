"""Residual Vector Quantizer for EnCodec."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class VectorQuantizer(nn.Module):
    """Single codebook vector quantizer.

    Maps continuous embeddings to nearest codebook entries.

    Attributes:
        codebook_size: Number of codebook entries
        codebook_dim: Dimension of each codebook vector
        embedding: Codebook embedding table
    """

    def __init__(
        self,
        codebook_size: int = 1024,
        codebook_dim: int = 128,
    ):
        """Initialize vector quantizer.

        Args:
            codebook_size: Number of codebook entries
            codebook_dim: Dimension of each codebook vector
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        # Codebook embedding table
        self.embedding = nn.Embedding(codebook_size, codebook_dim)

    def encode(self, x: mx.array) -> mx.array:
        """Encode continuous vectors to discrete codes.

        Args:
            x: Input embeddings [B, T, D]

        Returns:
            Codebook indices [B, T]
        """
        # Flatten batch and time dimensions
        shape = x.shape
        flat_x = x.reshape(-1, self.codebook_dim)  # [B*T, D]

        # Compute distances to all codebook entries
        # ||x - e||^2 = ||x||^2 + ||e||^2 - 2*x.e
        codebook = self.embedding.weight  # [K, D]

        x_norm_sq = mx.sum(flat_x ** 2, axis=-1, keepdims=True)  # [B*T, 1]
        e_norm_sq = mx.sum(codebook ** 2, axis=-1, keepdims=True).T  # [1, K]
        dot_product = flat_x @ codebook.T  # [B*T, K]

        distances = x_norm_sq + e_norm_sq - 2 * dot_product  # [B*T, K]

        # Find nearest codebook entry
        indices = mx.argmin(distances, axis=-1)  # [B*T]

        # Reshape back to [B, T]
        return indices.reshape(shape[:-1])

    def decode(self, codes: mx.array) -> mx.array:
        """Decode discrete codes to continuous vectors.

        Args:
            codes: Codebook indices [B, T] or [B, T, 1]

        Returns:
            Quantized embeddings [B, T, D]
        """
        if codes.ndim == 3:
            codes = codes.squeeze(-1)
        return self.embedding(codes)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Quantize input embeddings.

        Args:
            x: Input embeddings [B, T, D]

        Returns:
            Tuple of:
                - Quantized embeddings [B, T, D]
                - Codebook indices [B, T]
        """
        codes = self.encode(x)
        quantized = self.decode(codes)
        return quantized, codes


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer (RVQ) for multi-codebook quantization.

    RVQ applies multiple codebooks sequentially, where each codebook
    quantizes the residual from the previous codebook.

    Attributes:
        num_codebooks: Number of codebook layers
        codebook_size: Number of entries per codebook
        codebook_dim: Dimension of codebook vectors
        layers: List of VectorQuantizer modules
    """

    def __init__(
        self,
        num_codebooks: int = 4,
        codebook_size: int = 1024,
        codebook_dim: int = 128,
    ):
        """Initialize RVQ.

        Args:
            num_codebooks: Number of codebook layers
            codebook_size: Number of entries per codebook
            codebook_dim: Dimension of codebook vectors
        """
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        # Create quantizer for each codebook level
        self.layers = [
            VectorQuantizer(codebook_size, codebook_dim)
            for _ in range(num_codebooks)
        ]

    def encode(self, x: mx.array) -> mx.array:
        """Encode continuous embeddings to multi-codebook codes.

        Args:
            x: Input embeddings [B, T, D]

        Returns:
            Codebook indices [B, K, T] where K is num_codebooks
        """
        codes = []
        residual = x

        for layer in self.layers:
            indices = layer.encode(residual)
            codes.append(indices)
            quantized = layer.decode(indices)
            residual = residual - quantized

        # Stack codes: [K, B, T] -> [B, K, T]
        return mx.stack(codes, axis=0).transpose(1, 0, 2)

    def decode(self, codes: mx.array) -> mx.array:
        """Decode multi-codebook codes to continuous embeddings.

        Args:
            codes: Codebook indices [B, K, T] where K is num_codebooks

        Returns:
            Reconstructed embeddings [B, T, D]
        """
        # Transpose to [K, B, T] for easier iteration
        codes = codes.transpose(1, 0, 2)

        # Sum quantized outputs from all codebooks
        quantized = mx.zeros((codes.shape[1], codes.shape[2], self.codebook_dim))

        for i, layer in enumerate(self.layers):
            quantized = quantized + layer.decode(codes[i])

        return quantized

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Quantize input embeddings using RVQ.

        Args:
            x: Input embeddings [B, T, D]

        Returns:
            Tuple of:
                - Quantized embeddings [B, T, D]
                - Codebook indices [B, K, T]
        """
        codes = self.encode(x)
        quantized = self.decode(codes)
        return quantized, codes

    def get_codebook(self, layer_idx: int) -> mx.array:
        """Get codebook weights for a specific layer.

        Args:
            layer_idx: Index of the codebook layer

        Returns:
            Codebook weights [codebook_size, codebook_dim]
        """
        return self.layers[layer_idx].embedding.weight
