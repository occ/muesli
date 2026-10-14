import AVFoundation
import Foundation
import Testing
@testable import MuesliNativeApp

@Suite("Custom LLM dictation")
struct CustomLLMTranscriptionClientTests {
    @Test("request posts the WAV as input_audio with the language instruction and bearer key")
    func requestShape() throws {
        let audio = Data([0x52, 0x49, 0x46, 0x46])
        let request = try CustomLLMTranscriptionClient.request(
            audioData: audio,
            configuration: configuration(language: "Turkish")
        )

        #expect(request.url?.absoluteString == "https://litellm.example/v1/chat/completions")
        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer sk-test")
        #expect(request.value(forHTTPHeaderField: "x-team") == "dictation")

        let body = try #require(JSONSerialization.jsonObject(with: try #require(request.httpBody)) as? [String: Any])
        #expect(body["model"] as? String == "gemini-4-argon")
        #expect(body["max_tokens"] as? Int == CustomLLMTranscriptionClient.maxOutputTokens)
        let messages = try #require(body["messages"] as? [[String: Any]])
        #expect(messages.count == 2)
        #expect(messages[0]["role"] as? String == "system")
        #expect((messages[0]["content"] as? String)?.contains("The speech is in Turkish.") == true)
        #expect(messages[1]["role"] as? String == "user")
        let parts = try #require(messages[1]["content"] as? [[String: Any]])
        #expect(parts.first?["type"] as? String == "input_audio")
        let inputAudio = try #require(parts.first?["input_audio"] as? [String: Any])
        #expect(inputAudio["format"] as? String == "wav")
        #expect(inputAudio["data"] as? String == audio.base64EncodedString())
    }

    @Test("an empty language asks the model to keep the spoken language")
    func autoLanguage() {
        let prompt = CustomLLMTranscriptionClient.systemPrompt(language: "")
        #expect(prompt.contains("in the language that is spoken"))
        #expect(!prompt.contains("The speech is in"))
    }

    @Test("dictation is blocked until endpoint, OpenAI format, and model are set")
    func configurationGate() {
        var config = AppConfig()
        config.customLLMDictationModel = "gemini-4-argon"
        #expect(CustomLLMTranscriptionClient.configurationIssue(config: config) != nil)

        config.customLLMURL = "https://litellm.example"
        #expect(CustomLLMTranscriptionClient.configurationIssue(config: config) == nil)

        config.customLLMFormat = CustomLLMFormat.anthropic.rawValue
        #expect(CustomLLMTranscriptionClient.configurationIssue(config: config) != nil)

        config.customLLMFormat = CustomLLMFormat.openAI.rawValue
        config.customLLMDictationModel = "  "
        #expect(CustomLLMTranscriptionClient.configurationIssue(config: config) != nil)
    }

    @Test("a bare proxy host resolves to its chat completions endpoint")
    func endpointResolution() async throws {
        var config = AppConfig()
        config.customLLMURL = "https://litellm.example"
        config.customLLMDictationModel = " gemini-4-argon "
        config.customLLMDictationLanguage = " Turkish "
        config.customLLMAPIKey = " sk-test "

        let resolved = try await CustomLLMTranscriptionClient.configuration(from: config)
        #expect(resolved.requestURL.absoluteString == "https://litellm.example/v1/chat/completions")
        #expect(resolved.model == "gemini-4-argon")
        #expect(resolved.language == "Turkish")
        #expect(resolved.apiKey == "sk-test")
    }

    @Test("successful response returns the trimmed assistant content")
    func success() async throws {
        let text = try await transcribe(status: 200, body: #"{"choices":[{"message":{"content":"  Merhaba dünya \n"}}]}"#)
        #expect(text == "Merhaba dünya")
    }

    @Test("blank content is an empty transcript so the local fallback runs")
    func emptyTranscript() async {
        await #expect {
            try await transcribe(status: 200, body: #"{"choices":[{"message":{"content":"  "}}]}"#)
        } throws: { error in
            if case .emptyTranscript = error as? CustomLLMTranscriptionError { return true }
            return false
        }
    }

    @Test("HTTP errors surface the proxy's message")
    func serverError() async {
        await #expect {
            try await transcribe(status: 400, body: #"{"error":{"message":"model does not support audio"}}"#)
        } throws: { error in
            guard case .server(let status, let message) = error as? CustomLLMTranscriptionError else { return false }
            return status == 400 && message == "model does not support audio"
        }
    }

    @Test("silent recordings return no text without contacting the model")
    func silenceSkipsRequest() async throws {
        let calls = CustomLLMCallCounter()
        let text = try await transcribeTone(amplitude: 0.0005, calls: calls)
        #expect(text == "")
        #expect(await calls.count == 0)
    }

    @Test("quiet speech-level audio is still sent to the model")
    func quietAudioIsSent() async throws {
        let calls = CustomLLMCallCounter()
        let text = try await transcribeTone(amplitude: 0.02, calls: calls)
        #expect(text == "Merhaba")
        #expect(await calls.count == 1)
    }

    private func configuration(language: String = "") -> CustomLLMDictationConfiguration {
        CustomLLMDictationConfiguration(
            requestURL: URL(string: "https://litellm.example/v1/chat/completions")!,
            apiKey: "sk-test",
            extraHeaders: ["x-team": "dictation"],
            model: "gemini-4-argon",
            language: language
        )
    }

    private func transcribe(status: Int, body: String) async throws -> String {
        let wavURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("muesli-custom-llm-\(UUID().uuidString).wav")
        try Data([1, 2, 3]).write(to: wavURL)
        defer { try? FileManager.default.removeItem(at: wavURL) }
        let client = CustomLLMTranscriptionClient { request in
            let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!
            return (Data(body.utf8), response)
        }
        return try await client.transcribe(wavURL: wavURL, configuration: configuration())
    }

    /// Writes one second of a 220 Hz PCM16 tone at `amplitude` and transcribes it.
    private func transcribeTone(amplitude: Float, calls: CustomLLMCallCounter) async throws -> String {
        let wavURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("muesli-custom-llm-tone-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: wavURL) }
        let format = try #require(AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1))
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 16_000))
        buffer.frameLength = 16_000
        let samples = try #require(buffer.floatChannelData?[0])
        for index in 0..<16_000 {
            samples[index] = amplitude * sin(2 * .pi * 220 * Float(index) / 16_000)
        }
        do {
            let file = try AVAudioFile(forWriting: wavURL, settings: [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: 16_000,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false,
            ])
            try file.write(from: buffer)
        }
        let client = CustomLLMTranscriptionClient { request in
            await calls.increment()
            let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
            return (Data(#"{"choices":[{"message":{"content":"Merhaba"}}]}"#.utf8), response)
        }
        return try await client.transcribe(wavURL: wavURL, configuration: configuration())
    }
}

private actor CustomLLMCallCounter {
    private(set) var count = 0

    func increment() {
        count += 1
    }
}
