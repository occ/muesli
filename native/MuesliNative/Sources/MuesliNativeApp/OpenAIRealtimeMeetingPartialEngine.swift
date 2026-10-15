import Foundation

/// Live meeting transcript over one OpenAI Realtime transcription socket per
/// audio source. Muesli's VAD chunk boundaries become commits: `finish()`
/// commits the audio since the previous boundary and waits for that item's
/// final text, and `restart()` keeps the socket open for the next segment.
/// Partial text streams from the server's deltas while audio is still arriving.
actor OpenAIRealtimeMeetingPartialEngine: MeetingStreamingPartialEngine {
    nonisolated let partialDeliveryMode = MeetingStreamingPartialDeliveryMode.asynchronous

    /// The server rejects commits under 100 ms of audio (24 kHz PCM16).
    static let minimumCommitBytes = 4_800
    /// Reconnect at a segment boundary well before a provider session-length cap.
    static let maxSocketAge: Duration = .seconds(25 * 60)
    static let readyTimeout: Duration = .seconds(30)
    static let commitTimeout: Duration = .seconds(20)

    private let label: String
    private let resolveConfiguration: OpenAIRealtimeTranscriptionSession.ResolveConfiguration
    private var configuration: OpenAIDictationConfiguration?
    private var socket: URLSessionWebSocketTask?
    private var receiveTask: Task<Void, Never>?
    private var connectedAt: ContinuousClock.Instant?
    private var readyContinuation: CheckedContinuation<Void, Error>?
    private var commitContinuation: CheckedContinuation<String, Error>?
    private var timeoutTask: Task<Void, Never>?
    private var encoder = OpenAIRealtimePCMEncoder()
    private var uncommittedBytes = 0
    private var isAwaitingCommit = false
    private var committedItemID: String?
    private var textByItem: [String: String] = [:]
    private var completedTextByItem: [String: String] = [:]
    private var finalized: String?
    private var partialHandler: (@Sendable (String) -> Void)?
    private var failureHandler: (@Sendable (Error) -> Void)?

    init(label: String, resolveConfiguration: @escaping OpenAIRealtimeTranscriptionSession.ResolveConfiguration) {
        self.label = label
        self.resolveConfiguration = resolveConfiguration
    }

    func prepare() async throws {
        let configuration = try await resolveConfiguration()
        self.configuration = configuration
        let request = try OpenAIRealtimeProtocol.request(
            apiKey: configuration.apiKey,
            endpoint: configuration.endpoint,
            extraHeaders: configuration.extraHeaders
        )
        let socket = URLSession.shared.webSocketTask(with: request)
        self.socket = socket
        connectedAt = .now
        encoder = OpenAIRealtimePCMEncoder()
        uncommittedBytes = 0
        try await withCheckedThrowingContinuation { continuation in
            readyContinuation = continuation
            socket.resume()
            receiveTask = Task { [weak self] in await self?.receiveLoop(socket) }
            startTimeout(Self.readyTimeout)
        }
        fputs("[meeting-partials] \(label) realtime transcription session ready\n", stderr)
    }

    func setPartialHandler(_ handler: @escaping @Sendable (String) -> Void) async {
        partialHandler = handler
    }

    func setFailureHandler(_ handler: @escaping @Sendable (Error) -> Void) async {
        failureHandler = handler
    }

    func process(samples: [Float]) async throws {
        guard !samples.isEmpty else { return }
        guard let socket else { throw OpenAITranscriptionError.network(underlying: URLError(.networkConnectionLost)) }
        let pcm = encoder.encode(samples)
        guard !pcm.isEmpty else { return }
        uncommittedBytes += pcm.count
        try await socket.send(.string(try OpenAIRealtimeProtocol.append(pcm)))
    }

    func finish() async throws {
        finalized = nil
        guard let socket else { throw OpenAITranscriptionError.network(underlying: URLError(.networkConnectionLost)) }
        // Too little audio to commit; `restart()` clears it with the next segment.
        guard uncommittedBytes >= Self.minimumCommitBytes else {
            finalized = ""
            return
        }
        uncommittedBytes = 0
        committedItemID = nil
        isAwaitingCommit = true
        defer { isAwaitingCommit = false }
        try await socket.send(.string(OpenAIRealtimeProtocol.commit))
        let text = try await withCheckedThrowingContinuation { continuation in
            commitContinuation = continuation
            startTimeout(Self.commitTimeout)
            resolveCommitIfPossible()
        }
        finalized = text
        partialHandler?(text)
    }

    func finalizedText() async -> String? {
        finalized
    }

    func restart(
        partialHandler: @escaping @Sendable (String) -> Void,
        failureHandler: @escaping @Sendable (Error) -> Void
    ) async throws {
        self.partialHandler = partialHandler
        self.failureHandler = failureHandler
        finalized = nil
        if let socket, let connectedAt, ContinuousClock.now - connectedAt < Self.maxSocketAge {
            // Audio left on the server belongs to a dropped or too-short segment.
            if uncommittedBytes > 0 {
                try await socket.send(.string(OpenAIRealtimeProtocol.clear))
                uncommittedBytes = 0
            }
            encoder = OpenAIRealtimePCMEncoder()
            return
        }
        closeSocket()
        try await prepare()
    }

    func shutdown() async {
        closeSocket()
        timeoutTask?.cancel()
        let error = CancellationError()
        readyContinuation?.resume(throwing: error)
        readyContinuation = nil
        commitContinuation?.resume(throwing: error)
        commitContinuation = nil
        partialHandler = nil
        failureHandler = nil
        textByItem.removeAll()
        completedTextByItem.removeAll()
        fputs("[meeting-partials] \(label) realtime transcription session stopped\n", stderr)
    }

    private func receiveLoop(_ socket: URLSessionWebSocketTask) async {
        do {
            while !Task.isCancelled {
                let message = try await socket.receive()
                let string: String
                switch message {
                case .string(let value): string = value
                case .data(let data): string = String(decoding: data, as: UTF8.self)
                @unknown default: continue
                }
                try await handle(OpenAIRealtimeServerEvent(string), from: socket)
            }
        } catch {
            // A socket this engine already replaced or closed is not a failure.
            guard socket === self.socket, !Task.isCancelled else { return }
            fail(OpenAITranscriptionError.network(underlying: error))
        }
    }

    private func handle(_ event: OpenAIRealtimeServerEvent, from socket: URLSessionWebSocketTask) async throws {
        guard socket === self.socket else { return }
        switch event.type {
        case "session.created":
            guard let configuration else { return }
            try await socket.send(.string(try OpenAIRealtimeProtocol.sessionUpdate(
                model: configuration.model, language: configuration.language)))
        case "session.updated":
            timeoutTask?.cancel()
            readyContinuation?.resume()
            readyContinuation = nil
        case "input_audio_buffer.committed":
            guard isAwaitingCommit, committedItemID == nil else { return }
            committedItemID = event.itemID
            resolveCommitIfPossible()
        case "conversation.item.input_audio_transcription.delta":
            guard let itemID = event.itemID, let delta = event.transcript,
                  completedTextByItem[itemID] == nil else { return }
            textByItem[itemID, default: ""] += delta
            partialHandler?(textByItem[itemID] ?? "")
        case "conversation.item.input_audio_transcription.completed":
            guard let itemID = event.itemID else { return }
            completedTextByItem[itemID] = event.transcript ?? textByItem[itemID] ?? ""
            textByItem[itemID] = nil
            resolveCommitIfPossible()
        case "conversation.item.input_audio_transcription.failed", "error":
            fail(OpenAITranscriptionError.server(message: event.errorMessage ?? "Realtime transcription failed"))
        default:
            break
        }
    }

    private func resolveCommitIfPossible() {
        guard let continuation = commitContinuation,
              let itemID = committedItemID,
              let text = completedTextByItem.removeValue(forKey: itemID) else { return }
        commitContinuation = nil
        timeoutTask?.cancel()
        continuation.resume(returning: text.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    private func startTimeout(_ duration: Duration) {
        timeoutTask?.cancel()
        timeoutTask = Task { [weak self] in
            try? await Task.sleep(for: duration)
            guard !Task.isCancelled else { return }
            await self?.fail(OpenAITranscriptionError.timedOut)
        }
    }

    /// The meeting session goes dormant on failure and recovers from recorded audio.
    private func fail(_ error: Error) {
        timeoutTask?.cancel()
        closeSocket()
        readyContinuation?.resume(throwing: error)
        readyContinuation = nil
        commitContinuation?.resume(throwing: error)
        commitContinuation = nil
        failureHandler?(error)
    }

    private func closeSocket() {
        let socket = self.socket
        self.socket = nil
        connectedAt = nil
        receiveTask?.cancel()
        receiveTask = nil
        socket?.cancel(with: .goingAway, reason: nil)
    }
}
