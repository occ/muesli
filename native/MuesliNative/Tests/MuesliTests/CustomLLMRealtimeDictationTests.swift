import Foundation
import Testing
@testable import MuesliNativeApp

@Suite("Custom LLM realtime dictation")
struct CustomLLMRealtimeDictationTests {
    @Test("realtime socket is derived from any accepted Custom LLM endpoint form", arguments: [
        ("https://litellm.example", "wss://litellm.example/v1/realtime?intent=transcription"),
        ("https://litellm.example/v1", "wss://litellm.example/v1/realtime?intent=transcription"),
        ("https://litellm.example/v1/chat/completions", "wss://litellm.example/v1/realtime?intent=transcription"),
        ("https://gw.example/litellm", "wss://gw.example/litellm/v1/realtime?intent=transcription"),
        ("http://localhost:4000/v1", "ws://localhost:4000/v1/realtime?intent=transcription"),
    ])
    func realtimeURL(endpoint: String, expected: String) {
        var config = AppConfig()
        config.customLLMURL = endpoint
        #expect(CustomLLMRealtimeDictation.realtimeURL(config: config)?.absoluteString == expected)
    }

    @Test("dictation is blocked until endpoint, OpenAI format, and a credential are set")
    func configurationGate() {
        var config = AppConfig()
        config.customLLMAPIKeyCommand = "/usr/bin/true"
        #expect(CustomLLMRealtimeDictation.configurationIssue(config: config) != nil)
        #expect(CustomLLMRealtimeDictation.session(config: config) == nil)

        config.customLLMURL = "https://litellm.example"
        #expect(CustomLLMRealtimeDictation.configurationIssue(config: config) == nil)
        #expect(CustomLLMRealtimeDictation.session(config: config) != nil)

        config.customLLMFormat = CustomLLMFormat.anthropic.rawValue
        #expect(CustomLLMRealtimeDictation.configurationIssue(config: config) != nil)

        config.customLLMFormat = CustomLLMFormat.openAI.rawValue
        config.customLLMAPIKeyCommand = "  "
        #expect(CustomLLMRealtimeDictation.configurationIssue(config: config) != nil)
        config.customLLMAPIKey = "sk-test"
        #expect(CustomLLMRealtimeDictation.configurationIssue(config: config) == nil)
    }

    @Test("an empty dictation model falls back to the proxy's realtime transcription model")
    func defaultModel() {
        var config = AppConfig()
        #expect(CustomLLMRealtimeDictation.model(config: config) == "gpt-realtime-whisper")
        config.customLLMDictationModel = " custom-whisper "
        #expect(CustomLLMRealtimeDictation.model(config: config) == "custom-whisper")
    }

    @Test("session.update carries the language hint only when one is set")
    func languageHint() throws {
        #expect(try transcription(language: " tr ")["language"] as? String == "tr")
        #expect(try transcription(language: "")["language"] == nil)
    }

    @Test("custom headers reach the socket request but cannot replace the credential")
    func requestHeaders() throws {
        let endpoint = try #require(URL(string: "wss://litellm.example/v1/realtime?intent=transcription"))
        let request = try OpenAIRealtimeProtocol.request(
            apiKey: " sk-test ",
            endpoint: endpoint,
            extraHeaders: ["x-team": "dictation", "Authorization": "Bearer spoofed"]
        )
        #expect(request.url == endpoint)
        #expect(request.value(forHTTPHeaderField: "x-team") == "dictation")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer sk-test")
    }

    private func transcription(language: String) throws -> [String: Any] {
        let update = try OpenAIRealtimeProtocol.sessionUpdate(model: "gpt-realtime-whisper", language: language)
        let json = try #require(JSONSerialization.jsonObject(with: Data(update.utf8)) as? [String: Any])
        let session = try #require(json["session"] as? [String: Any])
        let audio = try #require(session["audio"] as? [String: Any])
        let input = try #require(audio["input"] as? [String: Any])
        return try #require(input["transcription"] as? [String: Any])
    }
}
