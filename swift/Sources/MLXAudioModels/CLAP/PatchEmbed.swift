// PatchEmbed.swift
// Patch embedding for HTSAT/Swin Transformer with optional fusion.
//
// Converts mel spectrograms to patch tokens using Conv2d,
// with optional attentional feature fusion for variable-length audio.

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - AFF Block

/// Attentional Feature Fusion block.
///
/// Fuses global and local features using learned attention weights.
/// Used when processing variable-length audio that requires multiple mel patches.
public class AFFBlock: Module, @unchecked Sendable {
    let channels: Int
    let reduced: Int

    // Local attention branch
    @ModuleInfo(key: "local_att_0") var localConv1: Conv2d
    @ModuleInfo(key: "local_att_1") var localBN1: BatchNorm
    @ModuleInfo(key: "local_att_3") var localConv2: Conv2d
    @ModuleInfo(key: "local_att_4") var localBN2: BatchNorm

    // Global attention branch
    @ModuleInfo(key: "global_att_0") var globalConv1: Conv2d
    @ModuleInfo(key: "global_att_1") var globalBN1: BatchNorm
    @ModuleInfo(key: "global_att_3") var globalConv2: Conv2d
    @ModuleInfo(key: "global_att_4") var globalBN2: BatchNorm

    public init(channels: Int, reduction: Int = 4) {
        self.channels = channels
        self.reduced = channels / reduction

        // Local attention: Conv1x1 -> BN -> ReLU -> Conv1x1 -> BN
        self._localConv1.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: reduced,
            kernelSize: 1
        )
        self._localBN1.wrappedValue = BatchNorm(featureCount: reduced)
        self._localConv2.wrappedValue = Conv2d(
            inputChannels: reduced,
            outputChannels: channels,
            kernelSize: 1
        )
        self._localBN2.wrappedValue = BatchNorm(featureCount: channels)

        // Global attention: similar structure
        self._globalConv1.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: reduced,
            kernelSize: 1
        )
        self._globalBN1.wrappedValue = BatchNorm(featureCount: reduced)
        self._globalConv2.wrappedValue = Conv2d(
            inputChannels: reduced,
            outputChannels: channels,
            kernelSize: 1
        )
        self._globalBN2.wrappedValue = BatchNorm(featureCount: channels)

        super.init()
    }

    /// Fuse hidden states with residual using attention.
    ///
    /// - Parameters:
    ///   - hiddenStates: Global features [B, H, W, C]
    ///   - residual: Local features [B, H, W, C]
    /// - Returns: Fused features [B, H, W, C]
    public func callAsFunction(_ hiddenStates: MLXArray, residual: MLXArray) -> MLXArray {
        let attentionInput = hiddenStates + residual

        // Local attention
        var localOut = localConv1(attentionInput)
        localOut = localBN1(localOut)
        localOut = relu(localOut)
        localOut = localConv2(localOut)
        localOut = localBN2(localOut)

        // Global attention (with global average pooling)
        // Mean over spatial dimensions, keeping dims
        var globalOut = MLX.mean(attentionInput, axes: [1, 2], keepDims: true)
        globalOut = globalConv1(globalOut)
        globalOut = globalBN1(globalOut)
        globalOut = relu(globalOut)
        globalOut = globalConv2(globalOut)
        globalOut = globalBN2(globalOut)

        // Combine with sigmoid
        let fusedWeight = sigmoid(localOut + globalOut)

        // Weighted fusion
        let output = 2 * hiddenStates * fusedWeight + 2 * residual * (1 - fusedWeight)
        return output
    }
}

// MARK: - Patch Embed

/// Patch embedding layer for audio spectrograms with optional fusion.
///
/// Converts mel spectrogram [B, C, H, W] to patch tokens [B, H*W, embedDim].
/// Uses Conv2d with kernel_size=patch_size and stride=patch_stride.
///
/// When enableFusion is true, handles multi-channel input [B, 4, H, W] where:
/// - Channel 0: Global mel spectrogram (downsampled to fit)
/// - Channels 1-3: Local mel patches for longer audio
public class PatchEmbed: Module, @unchecked Sendable {
    let patchSize: Int
    let patchStride: (Int, Int)
    let flatten: Bool
    let enableFusion: Bool

    @ModuleInfo var proj: Conv2d
    @ModuleInfo var norm: LayerNorm

    // Fusion components (optional)
    var melConv2d: Conv2d?
    var fusionModel: AFFBlock?

    public init(
        patchSize: Int = 4,
        patchStride: (Int, Int) = (4, 4),
        inChans: Int = 1,
        embedDim: Int = 96,
        flatten: Bool = true,
        enableFusion: Bool = false
    ) {
        self.patchSize = patchSize
        self.patchStride = patchStride
        self.flatten = flatten
        self.enableFusion = enableFusion

        // Main projection for global features
        // MLX Conv2d: inputChannels, outputChannels, kernelSize
        // Input is [B, H, W, C], so we project C -> embedDim
        self._proj.wrappedValue = Conv2d(
            inputChannels: inChans,
            outputChannels: embedDim,
            kernelSize: patchSize,
            stride: patchStride
        )
        self._norm.wrappedValue = LayerNorm(dimensions: embedDim)

        if enableFusion {
            // Conv for local mel patches
            // Kernel: [patchSize, patchSize * 3] = [4, 12]
            // Stride: [patchStride, patchStride * 3] = [4, 12]
            self.melConv2d = Conv2d(
                inputChannels: 1,
                outputChannels: embedDim,
                kernelSize: IntOrPair(patchSize, patchSize * 3),
                stride: IntOrPair(patchStride.0, patchStride.1 * 3)
            )
            self.fusionModel = AFFBlock(channels: embedDim)
        }

        super.init()
    }

    /// Forward pass.
    ///
    /// - Parameters:
    ///   - x: Input tensor [B, H, W, C] (mel spectrogram in NHWC format)
    ///        If enableFusion, expects C=4 with first channel as global
    ///   - isLongerIdx: Indices of samples that need fusion (have local features)
    /// - Returns: Patch embeddings [B, N, embedDim] if flatten else [B, H', W', embedDim]
    public func callAsFunction(_ x: MLXArray, isLongerIdx: [Int]? = nil) -> MLXArray {
        var output: MLXArray

        if enableFusion {
            // Input is [B, H, W, C] where C=4
            // Extract global features (first channel)
            let globalX = x[0..., 0..., 0..., 0..<1]  // [B, H, W, 1]

            var globalOut = proj(globalX)  // [B, H', W', embedDim]
            let outputWidth = globalOut.shape[2]

            // Handle fusion for longer samples
            if let indices = isLongerIdx, !indices.isEmpty {
                let idxArray = MLXArray(indices.map { Int32($0) })

                // Extract local features (channels 1-3) for longer samples
                // x[indices] -> [N, H, W, 4]
                let longerX = x.take(idxArray, axis: 0)
                let localX = longerX[0..., 0..., 0..., 1...]  // [N, H, W, 3]

                let localShape = localX.shape
                let BLocal = localShape[0]
                let H = localShape[1]
                let W = localShape[2]
                let numChannels = localShape[3]

                // Process each channel independently
                // Reshape to [N*3, H, W, 1]
                var localXReshaped = localX.transposed(axes: [0, 3, 1, 2])  // [N, 3, H, W]
                localXReshaped = localXReshaped.reshaped([BLocal * numChannels, H, W, 1])

                // Apply mel conv
                guard let melConv = melConv2d else {
                    fatalError("melConv2d not initialized for fusion")
                }
                var localOut = melConv(localXReshaped)  // [N*3, H', W', embedDim]

                let localOutShape = localOut.shape
                let localH = localOutShape[1]
                let localW = localOutShape[2]
                let embedDim = localOutShape[3]

                // Reshape back: [N, 3, H', W', embedDim]
                localOut = localOut.reshaped([BLocal, numChannels, localH, localW, embedDim])
                // Transpose to [N, H', 3, W', embedDim]
                localOut = localOut.transposed(axes: [0, 2, 1, 3, 4])
                // Flatten the 3 channels into width: [N, H', 3*W', embedDim]
                localOut = localOut.reshaped([BLocal, localH, -1, embedDim])

                let localWidth = localOut.shape[2]

                // Match widths
                if localWidth < outputWidth {
                    // Pad local to match global
                    let padWidth = outputWidth - localWidth
                    localOut = MLX.padded(localOut, widths: [(0, 0), (0, 0), (0, padWidth), (0, 0)])
                } else if localWidth > outputWidth {
                    // Crop local to match global
                    localOut = localOut[0..., 0..., 0..<outputWidth, 0...]
                }

                // Apply fusion
                let globalOutSubset = globalOut.take(idxArray, axis: 0)
                guard let fusion = fusionModel else {
                    fatalError("fusionModel not initialized")
                }
                let fused = fusion(globalOutSubset, residual: localOut)

                // Update globalOut with fused results
                let B = globalOut.shape[0]
                if indices.count == B {
                    // All samples get fused
                    globalOut = fused
                } else {
                    // Scatter update - create new array with fused values
                    // This is a simple implementation using concatenation
                    // For efficiency, could use more advanced indexing
                    var resultParts: [MLXArray] = []
                    var fusedIdx = 0
                    let indexSet = Set(indices)

                    for i in 0..<B {
                        if indexSet.contains(i) {
                            resultParts.append(fused[fusedIdx..<(fusedIdx + 1)])
                            fusedIdx += 1
                        } else {
                            resultParts.append(globalOut[i..<(i + 1)])
                        }
                    }
                    globalOut = MLX.concatenated(resultParts, axis: 0)
                }
            }

            output = globalOut
        } else {
            // Standard path without fusion
            output = proj(x)  // [B, H', W', embedDim]
        }

        if flatten {
            let shape = output.shape
            let B = shape[0]
            let H = shape[1]
            let W = shape[2]
            let C = shape[3]
            output = output.reshaped([B, H * W, C])  // [B, N, C]
        }

        output = norm(output)
        return output
    }
}
