// WhisperTokenizer.swift
// Whisper BPE tokenizer with special token handling.

import Foundation

/// Language codes supported by multilingual Whisper.
public let WHISPER_LANGUAGES: [String: String] = [
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
]

/// Whisper BPE tokenizer with special token handling.
///
/// This tokenizer handles:
/// - Standard BPE text encoding/decoding
/// - Special tokens for task specification (transcribe/translate)
/// - Language tokens for multilingual models
/// - Timestamp tokens for word-level alignment
/// - No-speech detection tokens
public class WhisperTokenizer {

    /// Vocabulary mapping token string to ID.
    private let vocab: [String: Int]

    /// Reverse vocabulary mapping ID to token string.
    private let reverseVocab: [Int: String]

    /// BPE merge pairs.
    private let merges: [(String, String)]

    /// BPE merge rankings for efficiency.
    private let bpeRanks: [String: Int]

    /// Special token IDs.
    private var specialTokens: [String: Int] = [:]

    /// Whether this is a multilingual tokenizer.
    public let isMultilingual: Bool

    /// Default language for encoding.
    public let language: String?

    /// Default task.
    public let task: String

    // MARK: - Special Token Properties

    /// End of text token ID.
    public var eot: Int { specialTokens["<|endoftext|>"] ?? 50256 }

    /// Start of transcript token ID.
    public var sot: Int { specialTokens["<|startoftranscript|>"] ?? 50257 }

    /// Translation task token ID.
    public var translate: Int { specialTokens["<|translate|>"] ?? 0 }

    /// Transcription task token ID.
    public var transcribe: Int { specialTokens["<|transcribe|>"] ?? 0 }

    /// No timestamps token ID.
    public var noTimestamps: Int { specialTokens["<|notimestamps|>"] ?? 0 }

    /// No speech token ID.
    public var noSpeech: Int { specialTokens["<|nospeech|>"] ?? 0 }

    /// First timestamp token ID (<|0.00|>).
    public var timestampBegin: Int { specialTokens["<|0.00|>"] ?? 0 }

    /// Last timestamp token ID (<|30.00|>).
    public var timestampEnd: Int { specialTokens["<|30.00|>"] ?? 0 }

    /// All language token IDs.
    public var allLanguageTokens: [Int] {
        guard isMultilingual else { return [] }
        return WhisperTokenizer.languages.keys.compactMap { lang in
            specialTokens["<|\(lang)|>"]
        }
    }

    /// Static access to languages dictionary.
    public static var languages: [String: String] { WHISPER_LANGUAGES }

    /// Reverse mapping from language name to code.
    public static var toLanguageCode: [String: String] {
        Dictionary(uniqueKeysWithValues: languages.map { ($0.value, $0.key) })
    }

    // MARK: - Initialization

    /// Initialize tokenizer from a HuggingFace tokenizer.json file.
    ///
    /// - Parameters:
    ///   - tokenizerPath: Path to tokenizer.json
    ///   - multilingual: Whether to use multilingual vocabulary
    ///   - language: Default language code
    ///   - task: Default task ("transcribe" or "translate")
    public init(
        tokenizerPath: URL,
        multilingual: Bool = true,
        language: String? = nil,
        task: String = "transcribe"
    ) throws {
        self.isMultilingual = multilingual
        self.language = language ?? (multilingual ? nil : "en")
        self.task = task

        // Load tokenizer.json
        let data = try Data(contentsOf: tokenizerPath)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            ?? [:]

        // Parse vocabulary from model.vocab
        var vocab: [String: Int] = [:]
        if let model = json["model"] as? [String: Any],
            let vocabDict = model["vocab"] as? [String: Int]
        {
            vocab = vocabDict
        }
        self.vocab = vocab
        self.reverseVocab = Dictionary(uniqueKeysWithValues: vocab.map { ($0.value, $0.key) })

        // Parse merges
        var merges: [(String, String)] = []
        var bpeRanks: [String: Int] = [:]
        if let model = json["model"] as? [String: Any],
            let mergesList = model["merges"] as? [String]
        {
            for (idx, merge) in mergesList.enumerated() {
                let parts = merge.split(separator: " ", maxSplits: 1)
                if parts.count == 2 {
                    let pair = (String(parts[0]), String(parts[1]))
                    merges.append(pair)
                    bpeRanks["\(pair.0) \(pair.1)"] = idx
                }
            }
        }
        self.merges = merges
        self.bpeRanks = bpeRanks

        // Parse added tokens (special tokens)
        if let addedTokens = json["added_tokens"] as? [[String: Any]] {
            for token in addedTokens {
                if let content = token["content"] as? String,
                    let id = token["id"] as? Int
                {
                    specialTokens[content] = id
                }
            }
        }

        // Initialize special tokens if not found in tokenizer.json
        initializeSpecialTokens()
    }

    /// Initialize with vocabulary and merges directly.
    ///
    /// - Parameters:
    ///   - vocab: Vocabulary mapping
    ///   - merges: BPE merge pairs
    ///   - multilingual: Whether multilingual
    ///   - language: Default language
    ///   - task: Default task
    public init(
        vocab: [String: Int],
        merges: [(String, String)],
        multilingual: Bool = true,
        language: String? = nil,
        task: String = "transcribe"
    ) {
        self.vocab = vocab
        self.reverseVocab = Dictionary(uniqueKeysWithValues: vocab.map { ($0.value, $0.key) })
        self.merges = merges
        self.bpeRanks = Dictionary(
            uniqueKeysWithValues: merges.enumerated().map {
                ("\($0.element.0) \($0.element.1)", $0.offset)
            })
        self.isMultilingual = multilingual
        self.language = language ?? (multilingual ? nil : "en")
        self.task = task

        initializeSpecialTokens()
    }

    /// Initialize special tokens based on Whisper's token layout.
    private func initializeSpecialTokens() {
        // Base vocabulary size
        let base = 50257

        var idx = base

        // End of text (same position as GPT-2)
        if specialTokens["<|endoftext|>"] == nil {
            specialTokens["<|endoftext|>"] = 50256
        }

        // Start of transcript
        if specialTokens["<|startoftranscript|>"] == nil {
            specialTokens["<|startoftranscript|>"] = idx
            idx += 1
        } else {
            idx = specialTokens["<|startoftranscript|>"]! + 1
        }

        // Language tokens (for multilingual)
        if isMultilingual {
            for langCode in WhisperTokenizer.languages.keys.sorted() {
                let token = "<|\(langCode)|>"
                if specialTokens[token] == nil {
                    specialTokens[token] = idx
                    idx += 1
                }
            }
        }

        // Task tokens
        if specialTokens["<|translate|>"] == nil {
            specialTokens["<|translate|>"] = idx
            idx += 1
        }
        if specialTokens["<|transcribe|>"] == nil {
            specialTokens["<|transcribe|>"] = idx
            idx += 1
        }

        // Additional special tokens
        for token in ["<|startoflm|>", "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>"] {
            if specialTokens[token] == nil {
                specialTokens[token] = idx
                idx += 1
            }
        }

        // Timestamp tokens: <|0.00|> through <|30.00|> in 0.02s increments
        for i in 0...1500 {
            let t = Float(i) * 0.02
            let token = String(format: "<|%.2f|>", t)
            if specialTokens[token] == nil {
                specialTokens[token] = idx
                idx += 1
            }
        }
    }

    // MARK: - Encoding

    /// Encode text to token IDs.
    ///
    /// - Parameter text: Text to encode
    /// - Returns: List of token IDs
    public func encode(_ text: String) -> [Int] {
        var tokens: [Int] = []

        // Simple word-level tokenization then BPE
        let words = tokenizeToWords(text)

        for word in words {
            let wordTokens = bpeEncode(word)
            tokens.append(contentsOf: wordTokens)
        }

        return tokens
    }

    /// Tokenize text into words for BPE processing.
    private func tokenizeToWords(_ text: String) -> [String] {
        // Simple whitespace-based tokenization with space preservation
        var words: [String] = []
        var currentWord = ""

        for char in text {
            if char.isWhitespace {
                if !currentWord.isEmpty {
                    words.append(currentWord)
                    currentWord = ""
                }
                // Add space as prefix to next word (GPT-2 style)
                currentWord = " "
            } else {
                currentWord.append(char)
            }
        }

        if !currentWord.isEmpty && currentWord != " " {
            words.append(currentWord)
        }

        return words
    }

    /// Apply BPE encoding to a word.
    private func bpeEncode(_ word: String) -> [Int] {
        // Convert to character-level tokens
        var pieces = word.map { String($0) }

        // Iteratively merge the most frequent pairs
        while pieces.count > 1 {
            // Find the best pair to merge
            var bestPair: (Int, String, String)?
            var bestRank = Int.max

            for i in 0..<(pieces.count - 1) {
                let pair = "\(pieces[i]) \(pieces[i + 1])"
                if let rank = bpeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestPair = (i, pieces[i], pieces[i + 1])
                }
            }

            // If no more merges possible, break
            guard let (idx, first, second) = bestPair else { break }

            // Apply the merge
            let merged = first + second
            pieces.remove(at: idx + 1)
            pieces[idx] = merged
        }

        // Convert pieces to token IDs
        return pieces.compactMap { vocab[$0] }
    }

    // MARK: - Decoding

    /// Decode token IDs to text.
    ///
    /// - Parameters:
    ///   - tokens: List of token IDs
    ///   - skipSpecialTokens: Whether to skip special tokens in output
    /// - Returns: Decoded text
    public func decode(_ tokens: [Int], skipSpecialTokens: Bool = true) -> String {
        var result = ""

        let specialIds = Set(specialTokens.values)

        for token in tokens {
            // Skip special tokens if requested
            if skipSpecialTokens && specialIds.contains(token) {
                continue
            }

            if let text = reverseVocab[token] {
                result += text
            }
        }

        // Clean up the result (handle byte-level encoding artifacts if present)
        return cleanDecodedText(result)
    }

    /// Clean decoded text by handling special encoding.
    private func cleanDecodedText(_ text: String) -> String {
        // Handle common GPT-2 byte encoding patterns
        var result = text

        // Replace common special characters
        result = result.replacingOccurrences(of: "Ġ", with: " ")
        result = result.replacingOccurrences(of: "Ċ", with: "\n")

        return result.trimmingCharacters(in: .whitespaces)
    }

    // MARK: - Language Tokens

    /// Get token ID for a language.
    ///
    /// - Parameter language: Language code (e.g., "en") or name (e.g., "english")
    /// - Returns: Token ID for the language
    public func getLanguageToken(_ language: String) throws -> Int {
        guard isMultilingual else {
            throw WhisperError.tokenizerError("Language tokens not available for English-only models")
        }

        // Convert language name to code if needed
        var langCode = language.lowercased()
        if let code = WhisperTokenizer.toLanguageCode[langCode] {
            langCode = code
        }

        guard WhisperTokenizer.languages[langCode] != nil else {
            throw WhisperError.tokenizerError(
                "Unknown language: '\(language)'. Supported: \(WhisperTokenizer.languages.keys.sorted())"
            )
        }

        guard let tokenId = specialTokens["<|\(langCode)|>"] else {
            throw WhisperError.tokenizerError("Language token not found for '\(langCode)'")
        }

        return tokenId
    }

    // MARK: - Initial Tokens

    /// Get initial token sequence for decoding.
    ///
    /// - Parameters:
    ///   - language: Language code (nil for auto-detect)
    ///   - task: Task type ("transcribe" or "translate")
    ///   - timestamps: Whether to include timestamps
    /// - Returns: List of initial token IDs
    public func getInitialTokens(
        language: String? = nil,
        task: String? = nil,
        timestamps: Bool = true
    ) -> [Int] {
        let task = task ?? self.task
        var tokens = [sot]

        // Add language token for multilingual
        if isMultilingual {
            if let language = language ?? self.language {
                if let langToken = try? getLanguageToken(language) {
                    tokens.append(langToken)
                }
            }
            // If nil, language will be detected during decoding
        }

        // Add task token
        if task == "translate" {
            tokens.append(translate)
        } else {
            tokens.append(transcribe)
        }

        // Add no-timestamps token if disabled
        if !timestamps {
            tokens.append(noTimestamps)
        }

        return tokens
    }

    // MARK: - Timestamp Tokens

    /// Check if a token is a timestamp token.
    public func isTimestamp(_ token: Int) -> Bool {
        token >= timestampBegin && token <= timestampEnd
    }

    /// Convert a timestamp token to seconds.
    ///
    /// - Parameter token: Timestamp token ID
    /// - Returns: Time in seconds
    public func timestampToSeconds(_ token: Int) throws -> Float {
        guard isTimestamp(token) else {
            throw WhisperError.tokenizerError("Token \(token) is not a timestamp token")
        }
        return Float(token - timestampBegin) * 0.02
    }

    /// Convert seconds to timestamp token.
    ///
    /// - Parameter seconds: Time in seconds (0.0 to 30.0)
    /// - Returns: Timestamp token ID
    public func secondsToTimestamp(_ seconds: Float) -> Int {
        let clampedSeconds = max(0.0, min(30.0, seconds))
        let idx = Int(round(clampedSeconds / 0.02))
        return timestampBegin + idx
    }
}
