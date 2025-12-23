// Tokenizer.swift
// BPE Tokenizer for CLAP text encoder.
//
// Implements byte-level BPE tokenization compatible with RoBERTa/GPT-2
// for encoding text into token IDs for the CLAP text encoder.

import Foundation

// MARK: - CLAP Tokenizer

/// BPE tokenizer for CLAP text encoding.
///
/// Uses byte-level BPE (like GPT-2/RoBERTa) to convert text to token IDs.
/// Handles special tokens, padding, and truncation for CLAP compatibility.
public class CLAPTokenizer: @unchecked Sendable {
    /// Vocabulary mapping tokens to IDs.
    public let vocab: [String: Int]

    /// Reverse vocabulary mapping IDs to tokens.
    public let idToToken: [Int: String]

    /// BPE merge rules (pairs and their priority).
    public let merges: [(String, String)]

    /// Merge priority lookup.
    private let mergeRanks: [String: Int]

    /// Byte to unicode mapping for byte-level BPE.
    private let byteEncoder: [UInt8: Character]
    private let byteDecoder: [Character: UInt8]

    /// Special token IDs.
    public let bosTokenId: Int  // <s>
    public let eosTokenId: Int  // </s>
    public let padTokenId: Int  // <pad>
    public let unkTokenId: Int  // <unk>
    public let maskTokenId: Int  // <mask>

    /// Maximum sequence length.
    public let maxLength: Int

    /// Creates a tokenizer with the given vocabulary and merges.
    public init(
        vocab: [String: Int],
        merges: [(String, String)],
        maxLength: Int = 77
    ) {
        self.vocab = vocab
        self.merges = merges
        self.maxLength = maxLength

        // Build reverse vocab
        var idToToken: [Int: String] = [:]
        for (token, id) in vocab {
            idToToken[id] = token
        }
        self.idToToken = idToToken

        // Build merge ranks
        var ranks: [String: Int] = [:]
        for (i, (a, b)) in merges.enumerated() {
            ranks["\(a) \(b)"] = i
        }
        self.mergeRanks = ranks

        // Build byte encoder/decoder
        var byteEnc: [UInt8: Character] = [:]
        var byteDec: [Character: UInt8] = [:]

        // Printable ASCII and extended characters
        var codepoints: [Int] = []
        codepoints.append(contentsOf: Array(33...126))  // ! to ~
        codepoints.append(contentsOf: Array(161...172))  // Extended Latin
        codepoints.append(contentsOf: Array(174...255))  // More extended

        var n = 0
        for b in 0..<256 {
            if !codepoints.contains(b) {
                codepoints.append(b)
                // Map to higher unicode range
                let char = Character(UnicodeScalar(256 + n)!)
                byteEnc[UInt8(b)] = char
                byteDec[char] = UInt8(b)
                n += 1
            } else {
                let char = Character(UnicodeScalar(b)!)
                byteEnc[UInt8(b)] = char
                byteDec[char] = UInt8(b)
            }
        }

        self.byteEncoder = byteEnc
        self.byteDecoder = byteDec

        // Get special token IDs
        self.bosTokenId = vocab["<s>"] ?? 0
        self.eosTokenId = vocab["</s>"] ?? 2
        self.padTokenId = vocab["<pad>"] ?? 1
        self.unkTokenId = vocab["<unk>"] ?? 3
        self.maskTokenId = vocab["<mask>"] ?? vocab.count - 1
    }

    /// Load tokenizer from vocab.json and merges.txt files.
    public static func load(vocabPath: URL, mergesPath: URL, maxLength: Int = 77) throws -> CLAPTokenizer {
        // Load vocabulary
        let vocabData = try Data(contentsOf: vocabPath)
        let vocab = try JSONDecoder().decode([String: Int].self, from: vocabData)

        // Load merges
        let mergesContent = try String(contentsOf: mergesPath, encoding: .utf8)
        let lines = mergesContent.components(separatedBy: .newlines)

        var merges: [(String, String)] = []
        for line in lines {
            // Skip header and empty lines
            if line.isEmpty || line.hasPrefix("#version") {
                continue
            }
            let parts = line.split(separator: " ", maxSplits: 1).map(String.init)
            if parts.count == 2 {
                merges.append((parts[0], parts[1]))
            }
        }

        return CLAPTokenizer(vocab: vocab, merges: merges, maxLength: maxLength)
    }

    /// Encode text to byte-level representation.
    private func bytesToUnicode(_ text: String) -> String {
        var result = ""
        for byte in text.utf8 {
            if let char = byteEncoder[byte] {
                result.append(char)
            }
        }
        return result
    }

    /// Get pairs of consecutive symbols in a word.
    private func getPairs(_ word: [String]) -> Set<String> {
        var pairs = Set<String>()
        for i in 0..<(word.count - 1) {
            pairs.insert("\(word[i]) \(word[i + 1])")
        }
        return pairs
    }

    /// Apply BPE to a single word.
    private func bpe(_ token: String) -> [String] {
        var word = token.map { String($0) }

        if word.count <= 1 {
            return word
        }

        var pairs = getPairs(word)

        while !pairs.isEmpty {
            // Find the lowest rank pair
            var minPair: String? = nil
            var minRank = Int.max

            for pair in pairs {
                if let rank = mergeRanks[pair], rank < minRank {
                    minRank = rank
                    minPair = pair
                }
            }

            guard let bestPair = minPair else { break }

            let parts = bestPair.split(separator: " ").map(String.init)
            let first = parts[0]
            let second = parts[1]

            // Merge the pair
            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == first && word[i + 1] == second {
                    newWord.append(first + second)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord

            if word.count == 1 {
                break
            }

            pairs = getPairs(word)
        }

        return word
    }

    /// Tokenize text into BPE tokens.
    private func tokenize(_ text: String) -> [String] {
        // Lowercase for CLAP
        let lowercased = text.lowercased()

        // Simple word splitting (basic regex-like tokenization)
        // Split on whitespace and punctuation while keeping them
        var tokens: [String] = []
        var currentWord = ""

        for char in lowercased {
            if char.isWhitespace {
                if !currentWord.isEmpty {
                    tokens.append(currentWord)
                    currentWord = ""
                }
                // Add space as its own token (for GPT-2 style)
                tokens.append(" ")
            } else if char.isPunctuation {
                if !currentWord.isEmpty {
                    tokens.append(currentWord)
                    currentWord = ""
                }
                tokens.append(String(char))
            } else {
                currentWord.append(char)
            }
        }
        if !currentWord.isEmpty {
            tokens.append(currentWord)
        }

        // Apply BPE to each token
        var bpeTokens: [String] = []
        for token in tokens {
            let encoded = bytesToUnicode(token)
            let subwords = bpe(encoded)
            bpeTokens.append(contentsOf: subwords)
        }

        return bpeTokens
    }

    /// Encode text to token IDs.
    ///
    /// - Parameters:
    ///   - text: Input text string
    ///   - addSpecialTokens: Whether to add BOS/EOS tokens
    ///   - padding: Pad to maxLength
    ///   - truncation: Truncate to maxLength
    /// - Returns: Array of token IDs
    public func encode(
        _ text: String,
        addSpecialTokens: Bool = true,
        padding: Bool = true,
        truncation: Bool = true
    ) -> [Int] {
        let tokens = tokenize(text)

        // Convert tokens to IDs
        var ids: [Int] = []
        for token in tokens {
            if let id = vocab[token] {
                ids.append(id)
            } else {
                ids.append(unkTokenId)
            }
        }

        // Add special tokens
        if addSpecialTokens {
            ids.insert(bosTokenId, at: 0)
            ids.append(eosTokenId)
        }

        // Truncate if needed
        if truncation && ids.count > maxLength {
            ids = Array(ids.prefix(maxLength))
            // Ensure EOS is at the end
            if addSpecialTokens {
                ids[ids.count - 1] = eosTokenId
            }
        }

        // Pad if needed
        if padding && ids.count < maxLength {
            let padCount = maxLength - ids.count
            ids.append(contentsOf: Array(repeating: padTokenId, count: padCount))
        }

        return ids
    }

    /// Encode text and return token IDs and attention mask.
    ///
    /// - Parameters:
    ///   - text: Input text string
    ///   - addSpecialTokens: Whether to add BOS/EOS tokens
    ///   - padding: Pad to maxLength
    ///   - truncation: Truncate to maxLength
    /// - Returns: Tuple of (inputIds, attentionMask)
    public func encodeWithMask(
        _ text: String,
        addSpecialTokens: Bool = true,
        padding: Bool = true,
        truncation: Bool = true
    ) -> (inputIds: [Int], attentionMask: [Int]) {
        let tokens = tokenize(text)

        // Convert tokens to IDs
        var ids: [Int] = []
        for token in tokens {
            if let id = vocab[token] {
                ids.append(id)
            } else {
                ids.append(unkTokenId)
            }
        }

        // Add special tokens
        if addSpecialTokens {
            ids.insert(bosTokenId, at: 0)
            ids.append(eosTokenId)
        }

        // Truncate if needed
        if truncation && ids.count > maxLength {
            ids = Array(ids.prefix(maxLength))
            if addSpecialTokens {
                ids[ids.count - 1] = eosTokenId
            }
        }

        // Create attention mask before padding
        var mask = Array(repeating: 1, count: ids.count)

        // Pad if needed
        if padding && ids.count < maxLength {
            let padCount = maxLength - ids.count
            ids.append(contentsOf: Array(repeating: padTokenId, count: padCount))
            mask.append(contentsOf: Array(repeating: 0, count: padCount))
        }

        return (ids, mask)
    }

    /// Decode token IDs back to text.
    public func decode(_ ids: [Int], skipSpecialTokens: Bool = true) -> String {
        var tokens: [String] = []

        for id in ids {
            if skipSpecialTokens {
                if id == bosTokenId || id == eosTokenId || id == padTokenId {
                    continue
                }
            }
            if let token = idToToken[id] {
                tokens.append(token)
            }
        }

        // Join and decode from byte representation
        let joined = tokens.joined()
        var bytes: [UInt8] = []
        for char in joined {
            if let byte = byteDecoder[char] {
                bytes.append(byte)
            }
        }

        return String(bytes: bytes, encoding: .utf8) ?? ""
    }
}

// MARK: - Default Tokenizer

extension CLAPTokenizer {
    /// Create a simple tokenizer with basic vocabulary.
    ///
    /// Note: For production use, load from actual vocab.json and merges.txt files
    /// from the CLAP model repository.
    public static func createBasic() -> CLAPTokenizer {
        // Minimal vocabulary for testing
        // In production, load from actual model files
        var vocab: [String: Int] = [
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "<unk>": 3,
            "<mask>": 50264,
        ]

        // Add basic ASCII characters
        for i in 0..<256 {
            let char = String(UnicodeScalar(i)!)
            if vocab[char] == nil {
                vocab[char] = vocab.count
            }
        }

        // Add common words
        let commonWords = [
            "the", "a", "an", "is", "are", "was", "were",
            "sound", "audio", "music", "noise", "speech",
            "dog", "cat", "bird", "car", "rain", "thunder",
            "barking", "meowing", "singing", "playing",
            "Ġ",  // Space prefix for GPT-2 style
        ]
        for word in commonWords {
            if vocab[word] == nil {
                vocab[word] = vocab.count
            }
            // Also add with space prefix
            let spaced = "Ġ" + word
            if vocab[spaced] == nil {
                vocab[spaced] = vocab.count
            }
        }

        // Basic merges (simplified)
        let merges: [(String, String)] = [
            ("t", "h"),
            ("th", "e"),
            ("Ġ", "a"),
            ("Ġ", "the"),
            ("s", "o"),
            ("so", "u"),
            ("sou", "nd"),
        ]

        return CLAPTokenizer(vocab: vocab, merges: merges)
    }
}
