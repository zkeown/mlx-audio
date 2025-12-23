// AudioEncoder.swift
// HTSAT (Hierarchical Token-Semantic Audio Transformer) audio encoder for CLAP.
//
// A Swin Transformer-based audio encoder that processes mel spectrograms
// into fixed-size embeddings for audio-text contrastive learning.

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - 2D Interpolation

/// Compute cubic interpolation weights using Keys kernel (a=-0.5).
private func cubicInterpWeights(_ t: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
    let a: Float = -0.5
    let t2 = t * t
    let t3 = t2 * t

    let w0 = a * t3 - 2 * a * t2 + a * t
    let w1 = (a + 2) * t3 - (a + 3) * t2 + 1
    let w2 = -(a + 2) * t3 + (2 * a + 3) * t2 - a * t
    let w3 = -a * t3 + a * t2

    return (w0, w1, w2, w3)
}

/// Interpolate 2D tensor to target shape.
///
/// - Parameters:
///   - x: Input tensor [B, C, H, W] (in NCHW format for compatibility)
///   - targetShape: Target (H, W)
///   - mode: Interpolation mode ("bicubic" or "bilinear")
///   - alignCorners: If true, corner pixels are aligned
/// - Returns: Interpolated tensor [B, C, targetH, targetW]
func interpolate2D(
    _ x: MLXArray,
    targetShape: (Int, Int),
    mode: String = "bicubic",
    alignCorners: Bool = true
) -> MLXArray {
    let shape = x.shape
    let B = shape[0]
    let C = shape[1]
    let H = shape[2]
    let W = shape[3]
    let (targetH, targetW) = targetShape

    if H == targetH && W == targetW {
        return x
    }

    // Create coordinate mapping
    let yCoords: MLXArray
    let xCoords: MLXArray

    if alignCorners {
        if targetH > 1 {
            yCoords = MLX.linspace(Float(0), Float(H - 1), count: targetH)
        } else {
            yCoords = MLXArray([Float(0)])
        }
        if targetW > 1 {
            xCoords = MLX.linspace(Float(0), Float(W - 1), count: targetW)
        } else {
            xCoords = MLXArray([Float(0)])
        }
    } else {
        let scaleH = Float(H) / Float(targetH)
        let scaleW = Float(W) / Float(targetW)
        yCoords = (MLXArray(0..<targetH).asType(.float32) + 0.5) * scaleH - 0.5
        xCoords = (MLXArray(0..<targetW).asType(.float32) + 0.5) * scaleW - 0.5
    }

    if mode == "bilinear" {
        // Bilinear interpolation
        let y0 = MLX.floor(yCoords).asType(.int32)
        let y1 = MLX.minimum(y0 + 1, MLXArray(Int32(H - 1)))
        let x0 = MLX.floor(xCoords).asType(.int32)
        let x1 = MLX.minimum(x0 + 1, MLXArray(Int32(W - 1)))

        let wy = (yCoords - y0.asType(.float32)).reshaped([1, 1, targetH, 1])
        let wx = (xCoords - x0.asType(.float32)).reshaped([1, 1, 1, targetW])

        // Sample values at corners
        // x[:, :, y0, :][:, :, :, x0]
        let val00 = x[0..., 0..., y0, 0...][0..., 0..., 0..., x0]
        let val01 = x[0..., 0..., y0, 0...][0..., 0..., 0..., x1]
        let val10 = x[0..., 0..., y1, 0...][0..., 0..., 0..., x0]
        let val11 = x[0..., 0..., y1, 0...][0..., 0..., 0..., x1]

        let result = val00 * (1 - wy) * (1 - wx) +
                     val01 * (1 - wy) * wx +
                     val10 * wy * (1 - wx) +
                     val11 * wy * wx

        return result
    } else {
        // Bicubic interpolation
        let yFloor = MLX.floor(yCoords).asType(.int32)
        let xFloor = MLX.floor(xCoords).asType(.int32)

        let ty = yCoords - yFloor.asType(.float32)
        let tx = xCoords - xFloor.asType(.float32)

        // Get 4 y and x indices (clamped)
        var yIndices: [MLXArray] = []
        var xIndices: [MLXArray] = []
        for offset in [-1, 0, 1, 2] {
            yIndices.append(MLX.clip(yFloor + Int32(offset), min: 0, max: Int32(H - 1)))
            xIndices.append(MLX.clip(xFloor + Int32(offset), min: 0, max: Int32(W - 1)))
        }

        // Compute cubic weights
        let (wy0, wy1, wy2, wy3) = cubicInterpWeights(ty)
        let (wx0, wx1, wx2, wx3) = cubicInterpWeights(tx)

        let wyList = [
            wy0.reshaped([1, 1, targetH, 1]),
            wy1.reshaped([1, 1, targetH, 1]),
            wy2.reshaped([1, 1, targetH, 1]),
            wy3.reshaped([1, 1, targetH, 1])
        ]
        let wxList = [
            wx0.reshaped([1, 1, 1, targetW]),
            wx1.reshaped([1, 1, 1, targetW]),
            wx2.reshaped([1, 1, 1, targetW]),
            wx3.reshaped([1, 1, 1, targetW])
        ]

        // Sample and interpolate
        var result = MLX.zeros([B, C, targetH, targetW])
        for (i, yi) in yIndices.enumerated() {
            for (j, xj) in xIndices.enumerated() {
                let vals = x[0..., 0..., yi, 0...][0..., 0..., 0..., xj]
                result = result + vals * wyList[i] * wxList[j]
            }
        }

        return result
    }
}

// MARK: - HTSAT

/// Hierarchical Token-Semantic Audio Transformer.
///
/// A Swin Transformer-based audio encoder that processes mel spectrograms
/// into fixed-size embeddings.
///
/// Architecture:
/// - Input: Log-mel spectrogram [B, C, F, T] (C=1 or 4 for fusion)
/// - BatchNorm (on frequency axis)
/// - reshapeMel2Img -> [B, C, 256, 256]
/// - PatchEmbed (with optional fusion) -> [B, N, embedDim]
/// - 4 stages of Swin Transformer blocks
/// - LayerNorm -> Reshape -> Grouped pooling
/// - Output: [B, hiddenSize]
public class HTSAT: Module, @unchecked Sendable {
    let config: CLAPAudioConfig
    let enableFusion: Bool
    let freqRatio: Int
    let specSize: Int

    @ModuleInfo(key: "batch_norm") var batchNorm: BatchNorm
    @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
    @ModuleInfo(key: "layers") var layers: [BasicLayer]
    @ModuleInfo var norm: LayerNorm

    /// Final output dimension.
    public let finalDim: Int

    public init(config: CLAPAudioConfig) {
        self.config = config
        self.enableFusion = config.enableFusion
        self.freqRatio = config.freqRatio
        self.specSize = config.specSize

        // Batch normalization on frequency dimension (nMels)
        self._batchNorm.wrappedValue = BatchNorm(featureCount: config.nMels)

        // Patch embedding with optional fusion
        self._patchEmbed.wrappedValue = PatchEmbed(
            patchSize: config.patchSize,
            patchStride: config.patchStride,
            inChans: 1,
            embedDim: config.embedDim,
            flatten: false,  // Keep spatial dims for Swin
            enableFusion: config.enableFusion
        )

        // Stochastic depth decay rule
        let numLayers = config.depths.reduce(0, +)
        var dpr: [Float] = []
        for i in 0..<numLayers {
            dpr.append(config.dropPathRate * Float(i) / Float(numLayers - 1))
        }

        // Build stages
        var layerList: [BasicLayer] = []
        var dim = config.embedDim
        var dprIdx = 0

        for i in 0..<config.depths.count {
            let depth = config.depths[i]
            let numHeads = config.numHeads[i]
            let dropPathRates = Array(dpr[dprIdx..<(dprIdx + depth)])

            let layer = BasicLayer(
                dim: dim,
                depth: depth,
                numHeads: numHeads,
                windowSize: config.windowSize,
                mlpRatio: config.mlpRatio,
                qkvBias: config.qkvBias,
                drop: config.dropRate,
                attnDrop: config.attnDropRate,
                dropPath: dropPathRates,
                downsampleEnabled: i < config.depths.count - 1  // No downsample on last stage
            )
            layerList.append(layer)
            dprIdx += depth

            // Update dim for next stage (doubled by patch merging)
            if i < config.depths.count - 1 {
                dim *= 2
            }
        }
        self._layers.wrappedValue = layerList
        self.finalDim = dim

        self._norm.wrappedValue = LayerNorm(dimensions: dim)

        super.init()
    }

    /// Reshape mel spectrogram to image format for Swin Transformer.
    ///
    /// Transforms input from [B, C, T, F] to [B, C, 256, 256] by:
    /// 1. Interpolating time dimension to specSize * freqRatio (1024)
    /// 2. Reshaping to reorganize frequency bins
    ///
    /// - Parameter x: Input tensor [B, C, T, F] where T=time, F=freq (nMels)
    /// - Returns: Reshaped tensor [B, C, 256, 256]
    func reshapeMel2Img(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let batch = shape[0]
        let channels = shape[1]
        var timeLength = shape[2]
        var freqLength = shape[3]

        let specWidth = specSize * freqRatio  // 256 * 4 = 1024
        let specHeight = specSize / freqRatio  // 256 / 4 = 64

        var out = x

        // Interpolate time dimension if needed
        if timeLength < specWidth {
            out = interpolate2D(out, targetShape: (specWidth, freqLength))
            timeLength = specWidth
        }

        // Interpolate freq dimension if needed
        if freqLength < specHeight {
            out = interpolate2D(out, targetShape: (out.shape[2], specHeight))
            freqLength = specHeight
        }

        let time = out.shape[2]
        let freq = out.shape[3]

        // Reshape: [B, C, T, F] -> [B, C*freqRatio, T/freqRatio, F]
        out = out.reshaped([batch, channels * freqRatio, time / freqRatio, freq])

        // Permute: -> [B, C*freqRatio, F, T/freqRatio]
        out = out.transposed(axes: [0, 1, 3, 2])

        // Reshape: -> [B, C, F*freqRatio, T/freqRatio]
        out = out.reshaped([batch, channels, freq * freqRatio, time / freqRatio])

        return out
    }

    /// Extract features before final projection.
    ///
    /// - Parameters:
    ///   - x: Input mel spectrogram [B, C, F, T] where C=1 or 4 for fusion
    ///   - isLonger: Boolean tensor [B] indicating which samples need fusion
    /// - Returns: Features [B, hiddenSize] before projection
    public func forwardFeatures(_ x: MLXArray, isLonger: MLXArray? = nil) -> MLXArray {
        let B = x.shape[0]

        // Input: [B, C, F, T] -> convert to [B, C, T, F] for consistent processing
        var out = x.transposed(axes: [0, 1, 3, 2])

        // Apply batch norm on frequency axis (last dim)
        // BatchNorm expects [B, H, W, C] or [B, ..., C]
        // Our input is [B, C, T, F] - need to normalize over F
        // Transpose to [B, T, F, C], apply BN, transpose back
        out = out.transposed(axes: [0, 2, 3, 1])  // [B, T, F, C]

        // For single channel, we need [B, T, F] for BN which normalizes last dim
        if out.shape[3] == 1 {
            out = out.squeezed(axis: 3)  // [B, T, F]
            out = batchNorm(out)
            out = out.expandedDimensions(axis: 3)  // [B, T, F, 1]
        } else {
            // Multi-channel case - more complex, skip for now
            // Apply BN per channel
            out = batchNorm(out)
        }

        out = out.transposed(axes: [0, 3, 1, 2])  // Back to [B, C, T, F]

        // Reshape mel to image format
        out = reshapeMel2Img(out)  // [B, C, 256, 256]

        // Convert to NHWC for MLX Conv2d
        out = out.transposed(axes: [0, 2, 3, 1])  // [B, H, W, C]

        // Determine fusion indices
        var isLongerIdx: [Int]? = nil
        if enableFusion, let longer = isLonger {
            // Find indices where isLonger is true
            let longerFlat = longer.reshaped([-1])
            var indices: [Int] = []
            let values = longerFlat.asArray(Float.self)
            for (i, v) in values.enumerated() {
                if v > 0.5 {
                    indices.append(i)
                }
            }
            if !indices.isEmpty {
                isLongerIdx = indices
            }
        }

        // Patch embedding (keeps spatial dims)
        out = patchEmbed(out, isLongerIdx: isLongerIdx)  // [B, H, W, C] or [B, L, C]

        // If patchEmbed didn't flatten, do it now
        if out.ndim == 4 {
            let s = out.shape
            out = out.reshaped([s[0], s[1] * s[2], s[3]])  // [B, L, C]
        }

        var H = specSize / config.patchStride.0  // 256 / 4 = 64
        var W = specSize / config.patchStride.1  // 256 / 4 = 64

        // Pass through Swin stages
        for layer in layers {
            (out, H, W) = layer(out, H: H, W: W)
        }

        // Final norm
        out = norm(out)

        // Reshape and pool using grouped pooling
        let nChannels = out.shape[2]

        // Reshape to [B, H, W, C] then to [B, C, H, W] for pooling
        out = out.reshaped([B, H, W, nChannels])
        out = out.transposed(axes: [0, 3, 1, 2])  // [B, C, H, W]

        // Grouped pooling
        let nFrequencies = out.shape[2]  // H
        let nTemp = out.shape[3]  // W
        let cFreqBin = nFrequencies / freqRatio

        // Reshape: [B, C, nFreq/cFreqBin, cFreqBin, nTemp]
        out = out.reshaped([B, nChannels, nFrequencies / cFreqBin, cFreqBin, nTemp])

        // Permute: -> [B, C, cFreqBin, nFreq/cFreqBin, nTemp]
        out = out.transposed(axes: [0, 1, 3, 2, 4])

        // Reshape: -> [B, C, cFreqBin, (nFreq/cFreqBin)*nTemp]
        out = out.reshaped([B, nChannels, cFreqBin, -1])

        // Flatten from dim 2: -> [B, C, flat]
        out = out.reshaped([B, nChannels, -1])

        // Adaptive average pooling: mean over last dim -> [B, C]
        out = MLX.mean(out, axis: -1)

        return out
    }

    public func callAsFunction(_ x: MLXArray, isLonger: MLXArray? = nil) -> MLXArray {
        return forwardFeatures(x, isLonger: isLonger)
    }
}

// MARK: - Audio Fusion

/// Fusion module for variable-length audio.
///
/// Handles audio longer than the model's fixed input size by:
/// 1. Splitting audio into overlapping chunks
/// 2. Processing each chunk independently
/// 3. Aggregating chunk embeddings
public class AudioFusion: Module, @unchecked Sendable {
    let config: CLAPAudioConfig
    var fusionWeight: MLXArray?

    public init(config: CLAPAudioConfig) {
        self.config = config

        if config.fusionType == "aff_2d" {
            self.fusionWeight = MLX.ones([1])
        }

        super.init()
    }

    /// Process variable-length audio with fusion.
    ///
    /// - Parameters:
    ///   - mel: Mel spectrogram [B, 1, F, T]
    ///   - encoder: HTSAT encoder
    ///   - chunkSize: Size of each chunk (time frames)
    ///   - overlap: Overlap ratio between chunks
    /// - Returns: Fused embeddings [B, hiddenSize]
    public func callAsFunction(
        _ mel: MLXArray,
        encoder: HTSAT,
        chunkSize: Int = 1024,
        overlap: Float = 0.5
    ) -> MLXArray {
        let shape = mel.shape
        let B = shape[0]
        let T = shape[3]

        if T <= chunkSize {
            // No fusion needed
            return encoder(mel)
        }

        // Calculate chunk parameters
        let hop = Int(Float(chunkSize) * (1 - overlap))
        let numChunks = (T - chunkSize) / hop + 1

        // Process chunks
        var chunkEmbeddings: [MLXArray] = []
        for i in 0..<numChunks {
            let start = i * hop
            let end = start + chunkSize
            let chunk = mel[0..., 0..., 0..., start..<end]
            let emb = encoder(chunk)
            chunkEmbeddings.append(emb.expandedDimensions(axis: 1))
        }

        // Stack and aggregate
        let stacked = MLX.concatenated(chunkEmbeddings, axis: 1)  // [B, numChunks, hidden]

        // Simple mean fusion
        let fused = MLX.mean(stacked, axis: 1)

        return fused
    }
}
