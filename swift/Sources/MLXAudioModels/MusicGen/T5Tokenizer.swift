// T5Tokenizer.swift
// SentencePiece-based tokenizer for T5 text encoder.

import Foundation
import MLX

/// T5 tokenizer using HuggingFace tokenizer.json format.
/// Implements SentencePiece unigram tokenization for T5.
public class T5Tokenizer {

    // MARK: - Vocabulary

    private var vocabulary: [String: Int]
    private var reverseVocabulary: [Int: String]
    private var merges: [(String, String)]

    // MARK: - Special Tokens

    public let padTokenId: Int
    public let eosTokenId: Int
    public let unkTokenId: Int

    public let padToken: String = "<pad>"
    public let eosToken: String = "</s>"
    public let unkToken: String = "<unk>"

    // MARK: - Configuration

    public let maxLength: Int

    // MARK: - Initialization

    public init(
        vocabulary: [String: Int],
        merges: [(String, String)] = [],
        padTokenId: Int = 0,
        eosTokenId: Int = 1,
        unkTokenId: Int = 2,
        maxLength: Int = 256
    ) {
        self.vocabulary = vocabulary
        self.merges = merges
        self.padTokenId = padTokenId
        self.eosTokenId = eosTokenId
        self.unkTokenId = unkTokenId
        self.maxLength = maxLength

        // Build reverse vocabulary
        self.reverseVocabulary = [:]
        for (token, id) in vocabulary {
            self.reverseVocabulary[id] = token
        }
    }

    /// Load tokenizer from HuggingFace tokenizer.json file.
    public static func fromPretrained(path: String) throws -> T5Tokenizer {
        let tokenizerPath = (path as NSString).appendingPathComponent("tokenizer.json")

        guard FileManager.default.fileExists(atPath: tokenizerPath) else {
            throw T5TokenizerError.fileNotFound(tokenizerPath)
        }

        let data = try Data(contentsOf: URL(fileURLWithPath: tokenizerPath))
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        guard let json = json else {
            throw T5TokenizerError.invalidFormat("Failed to parse tokenizer.json")
        }

        // Extract vocabulary from model.vocab
        var vocabulary: [String: Int] = [:]

        if let model = json["model"] as? [String: Any],
            let vocab = model["vocab"] as? [[Any]]
        {
            // Format: [[token, score], ...]
            for (index, item) in vocab.enumerated() {
                if let token = item.first as? String {
                    vocabulary[token] = index
                }
            }
        }

        // Extract special tokens
        var padId = 0
        var eosId = 1
        var unkId = 2

        if let addedTokens = json["added_tokens"] as? [[String: Any]] {
            for token in addedTokens {
                if let content = token["content"] as? String,
                    let id = token["id"] as? Int
                {
                    switch content {
                    case "<pad>": padId = id
                    case "</s>": eosId = id
                    case "<unk>": unkId = id
                    default: break
                    }
                    vocabulary[content] = id
                }
            }
        }

        return T5Tokenizer(
            vocabulary: vocabulary,
            padTokenId: padId,
            eosTokenId: eosId,
            unkTokenId: unkId
        )
    }

    // MARK: - Tokenization

    /// Encode text to token IDs.
    /// - Parameters:
    ///   - text: Input text string
    ///   - addSpecialTokens: Whether to add EOS token
    ///   - padding: Whether to pad to maxLength
    /// - Returns: Tuple of (inputIds, attentionMask) as MLXArrays
    public func encode(
        _ text: String,
        addSpecialTokens: Bool = true,
        padding: Bool = true
    ) -> (inputIds: MLXArray, attentionMask: MLXArray) {
        // Tokenize using SentencePiece-style algorithm
        var tokens = tokenize(text)

        // Add EOS token
        if addSpecialTokens {
            tokens.append(eosTokenId)
        }

        // Truncate if needed
        if tokens.count > maxLength {
            tokens = Array(tokens.prefix(maxLength))
        }

        // Create attention mask (1 for real tokens, 0 for padding)
        var attentionMask = Array(repeating: 1, count: tokens.count)

        // Pad to maxLength
        if padding {
            let paddingLength = maxLength - tokens.count
            if paddingLength > 0 {
                tokens.append(contentsOf: Array(repeating: padTokenId, count: paddingLength))
                attentionMask.append(contentsOf: Array(repeating: 0, count: paddingLength))
            }
        }

        return (
            inputIds: MLXArray(tokens),
            attentionMask: MLXArray(attentionMask)
        )
    }

    /// Batch encode multiple texts.
    public func encodeBatch(
        _ texts: [String],
        addSpecialTokens: Bool = true,
        padding: Bool = true
    ) -> (inputIds: MLXArray, attentionMask: MLXArray) {
        var inputIdArrays: [MLXArray] = []
        var maskArrays: [MLXArray] = []

        for text in texts {
            let (ids, mask) = encode(text, addSpecialTokens: addSpecialTokens, padding: padding)
            inputIdArrays.append(ids.expandedDimensions(axis: 0))
            maskArrays.append(mask.expandedDimensions(axis: 0))
        }

        return (
            inputIds: concatenated(inputIdArrays, axis: 0),
            attentionMask: concatenated(maskArrays, axis: 0)
        )
    }

    /// Decode token IDs back to text.
    public func decode(_ tokenIds: MLXArray, skipSpecialTokens: Bool = true) -> String {
        let ids = tokenIds.asArray(Int.self)
        var tokens: [String] = []

        for id in ids {
            // Skip special tokens if requested
            if skipSpecialTokens {
                if id == padTokenId || id == eosTokenId {
                    continue
                }
            }

            if let token = reverseVocabulary[id] {
                tokens.append(token)
            }
        }

        // Join tokens and clean up SentencePiece artifacts
        var text = tokens.joined()
        text = text.replacingOccurrences(of: "▁", with: " ")
        text = text.trimmingCharacters(in: .whitespaces)

        return text
    }

    // MARK: - Internal Tokenization

    /// Tokenize text using a simplified SentencePiece-like algorithm.
    private func tokenize(_ text: String) -> [Int] {
        // Preprocess: add space prefix (SentencePiece convention)
        let processedText = "▁" + text.replacingOccurrences(of: " ", with: "▁")

        // Greedy tokenization: find longest matching tokens
        var tokens: [Int] = []
        var start = processedText.startIndex

        while start < processedText.endIndex {
            var bestMatch: (String, Int)? = nil
            var end = processedText.endIndex

            // Find longest matching token from current position
            while end > start {
                let substring = String(processedText[start ..< end])
                if let tokenId = vocabulary[substring] {
                    bestMatch = (substring, tokenId)
                    break
                }
                end = processedText.index(before: end)
            }

            if let match = bestMatch {
                tokens.append(match.1)
                start = processedText.index(start, offsetBy: match.0.count)
            } else {
                // Unknown character - use UNK token and advance by one character
                tokens.append(unkTokenId)
                start = processedText.index(after: start)
            }
        }

        return tokens
    }
}

// MARK: - Errors

public enum T5TokenizerError: Error, LocalizedError {
    case fileNotFound(String)
    case invalidFormat(String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "Tokenizer file not found: \(path)"
        case .invalidFormat(let reason):
            return "Invalid tokenizer format: \(reason)"
        }
    }
}
