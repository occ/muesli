import Foundation
import MuesliCore

/// Dictation through the Custom LLM connection: OpenAI Realtime transcription
/// on the same OpenAI-compatible host (for example a LiteLLM proxy that serves
/// `/v1/realtime?intent=transcription`). Endpoint, API key, key command, and
/// headers are shared with Custom LLM summaries and cleanup; only the model and
/// language are dictation-specific.
enum CustomLLMRealtimeDictation {
    static let defaultModel = "gpt-realtime-whisper"
    static let backend = "custom-llm-realtime"

    /// Why dictation cannot start with `config`, or nil when it can. Checked
    /// before recording so the user is not left talking into a dead provider.
    static func configurationIssue(config: AppConfig) -> String? {
        if config.customLLMURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "Set the Custom LLM endpoint in Settings → Dictation."
        }
        guard (CustomLLMFormat(rawValue: config.customLLMFormat) ?? .openAI) == .openAI else {
            return "Custom LLM dictation needs the OpenAI-compatible API format."
        }
        guard realtimeURL(config: config) != nil else {
            return "Invalid Custom LLM endpoint: \(config.customLLMURL)"
        }
        if config.customLLMAPIKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           config.customLLMAPIKeyCommand.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "Enter a Custom LLM API key or API key command in Settings → Dictation."
        }
        if (try? CustomLLMRequestHeaders.validated(config.customLLMHeaders)) == nil {
            return "Fix the Custom LLM headers in Settings → Dictation."
        }
        return nil
    }

    /// The realtime socket on the Custom LLM host: the chat-completions endpoint
    /// with `chat/completions` replaced by `realtime`, over ws(s).
    static func realtimeURL(config: AppConfig) -> URL? {
        guard let chatURL = MeetingSummaryClient.resolveCustomLLMURL(config: config, format: .openAI),
              var components = URLComponents(url: chatURL, resolvingAgainstBaseURL: false) else {
            return nil
        }
        var pathParts = components.path.split(separator: "/").map(String.init)
        if pathParts.suffix(2) == ["chat", "completions"] { pathParts.removeLast(2) }
        components.path = "/" + (pathParts + ["realtime"]).joined(separator: "/")
        switch components.scheme?.lowercased() {
        case "https": components.scheme = "wss"
        case "http": components.scheme = "ws"
        default: return nil
        }
        // No `model` query item: the proxy routes intent=transcription to its
        // transcription deployment.
        components.queryItems = [URLQueryItem(name: "intent", value: "transcription")]
        return components.url
    }

    static func model(config: AppConfig) -> String {
        let trimmed = config.customLLMDictationModel.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? defaultModel : trimmed
    }

    /// A live session, or nil when `configurationIssue` would block dictation.
    /// Only Sendable values are captured; the API key command runs when the
    /// socket opens, off the hotkey path.
    static func session(config: AppConfig) -> OpenAIHostedDictationSession? {
        guard configurationIssue(config: config) == nil,
              let endpoint = realtimeURL(config: config),
              let headers = try? CustomLLMRequestHeaders.validated(config.customLLMHeaders) else {
            return nil
        }
        let model = model(config: config)
        let language = config.customLLMDictationLanguage.trimmingCharacters(in: .whitespacesAndNewlines)
        let keyCommand = config.customLLMAPIKeyCommand
        let savedKey = config.customLLMAPIKey
        return OpenAIHostedDictationSession(model: model, backend: backend, endpoint: endpoint) {
            OpenAIDictationConfiguration(
                apiKey: try await CredentialCommandRunner.resolve(command: keyCommand, fallback: savedKey),
                model: model,
                endpoint: endpoint,
                extraHeaders: headers,
                language: language
            )
        }
    }
}
