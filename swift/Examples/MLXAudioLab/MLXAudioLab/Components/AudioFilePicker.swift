// AudioFilePicker.swift
// Cross-platform audio file picker component.

import SwiftUI
import UniformTypeIdentifiers

/// Cross-platform audio file picker.
struct AudioFilePicker: View {
    @Binding var selectedURL: URL?
    var label: String = "Select Audio File"
    var allowedTypes: [UTType] = [.audio, .mpeg4Audio, .mp3, .wav, .aiff]

    #if os(iOS)
    @State private var showPicker = false
    #endif

    var body: some View {
        #if os(iOS)
        Button {
            showPicker = true
        } label: {
            HStack {
                Image(systemName: selectedURL == nil ? "doc.badge.plus" : "doc.fill")
                Text(selectedURL?.lastPathComponent ?? label)
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color.systemGray5)
            .cornerRadius(10)
        }
        .sheet(isPresented: $showPicker) {
            DocumentPicker(selectedURL: $selectedURL, allowedTypes: allowedTypes)
        }
        #else
        Button {
            selectFileOnMac()
        } label: {
            HStack {
                Image(systemName: selectedURL == nil ? "doc.badge.plus" : "doc.fill")
                Text(selectedURL?.lastPathComponent ?? label)
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color.gray.opacity(0.2))
            .cornerRadius(10)
        }
        #endif
    }

    #if os(macOS)
    private func selectFileOnMac() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = allowedTypes
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false

        if panel.runModal() == .OK {
            selectedURL = panel.url
        }
    }
    #endif
}

#if os(iOS)
/// iOS document picker wrapper.
struct DocumentPicker: UIViewControllerRepresentable {
    @Binding var selectedURL: URL?
    let allowedTypes: [UTType]

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: allowedTypes)
        picker.delegate = context.coordinator
        picker.allowsMultipleSelection = false
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let parent: DocumentPicker

        init(_ parent: DocumentPicker) {
            self.parent = parent
        }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            guard let url = urls.first else { return }

            // Start accessing security-scoped resource
            guard url.startAccessingSecurityScopedResource() else {
                return
            }

            parent.selectedURL = url
        }

        func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
            // User cancelled
        }
    }
}
#endif

/// Button to export/save audio files.
struct AudioExportButton: View {
    let audio: Data?
    let filename: String
    var label: String = "Export"

    #if os(iOS)
    @State private var showExporter = false
    #endif

    var body: some View {
        Button {
            #if os(iOS)
            showExporter = true
            #else
            exportOnMac()
            #endif
        } label: {
            Label(label, systemImage: "square.and.arrow.up")
        }
        .disabled(audio == nil)
        #if os(iOS)
        .fileExporter(
            isPresented: $showExporter,
            document: AudioDocument(data: audio ?? Data()),
            contentType: .wav,
            defaultFilename: filename
        ) { result in
            // Handle result
        }
        #endif
    }

    #if os(macOS)
    private func exportOnMac() {
        guard let audio = audio else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.wav]
        panel.nameFieldStringValue = filename

        if panel.runModal() == .OK, let url = panel.url {
            try? audio.write(to: url)
        }
    }
    #endif
}

#if os(iOS)
/// Document type for audio export.
struct AudioDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.wav, .audio] }

    var data: Data

    init(data: Data) {
        self.data = data
    }

    init(configuration: ReadConfiguration) throws {
        data = configuration.file.regularFileContents ?? Data()
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}
#endif

// MARK: - Preview

#Preview {
    VStack(spacing: 20) {
        AudioFilePicker(selectedURL: .constant(nil))

        AudioFilePicker(
            selectedURL: .constant(URL(fileURLWithPath: "/path/to/song.mp3")),
            label: "Choose Song"
        )
    }
    .padding()
}
