import whisper
import csv
from time import time
from pathlib import Path

data_dir = Path("recordings_stt")
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a"}


model = whisper.load_model("tiny", device="cuda")


def find_audio_files(root):
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def read_text(txt_path):
    try:
        return txt_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return txt_path.read_text(encoding="windows-1250").strip()


if __name__ == "__main__":
    csv_filename = "transcription_results.csv"

    audio_files = find_audio_files(data_dir)
    print(f"Znaleziono {len(audio_files)} plików audio")
    # for debugging
    print(*audio_files, sep="\n")
    # Debug: porównanie plików .txt z plikami audio
    audio_stems = {p.with_suffix("") for p in audio_files}
    txt_files = sorted(data_dir.rglob("*.txt"))
    txt_stems = {p.with_suffix("") for p in txt_files}

    print(f"\nPliki .txt: {len(txt_files)}")

    print("\n--- Jest .txt, ale nie znaleziono audio ---")
    for stem in sorted(txt_stems - audio_stems):
        print(stem)

    print("\n--- Jest audio, ale nie ma .txt ---")
    for stem in sorted(audio_stems - txt_stems):
        print(stem)

    # Wszystkie pliki z innymi rozszerzeniami
    print("\n--- Pliki z innym rozszerzeniem niż .wav/.mp3/.txt ---")
    for p in sorted(data_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() not in AUDIO_EXTENSIONS | {".txt"}:
            print(repr(str(p)))

    with open(csv_filename, mode="w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "Folder", "File Name", "Audio Duration (s)",
            "Transcription Time (s)", "Original Text", "Transcribed Text",
        ])

        for audio_path in audio_files:
            txt_path = audio_path.with_suffix(".txt")

            if txt_path.exists():
                original_text = read_text(txt_path)
            else:
                print(f"Warning: brak pliku .txt dla {audio_path.name}")
                original_text = ""

            print(f"Przetwarzanie: {audio_path.parent.name}/{audio_path.name}...")

            try:

                audio = whisper.load_audio(str(audio_path))
                duration = len(audio) / whisper.audio.SAMPLE_RATE

                start_time = time()
                result = model.transcribe(audio, language="pl")
                transcription_time = time() - start_time

                writer.writerow([
                    audio_path.parent.name,
                    audio_path.name,
                    f"{duration:.2f}",
                    f"{transcription_time:.2f}",
                    original_text,
                    result["text"].strip(),
                ])
                csv_file.flush()
                print(f"-> Zapisano wynik dla {audio_path.name}")

            except Exception as e:
                print(f"Błąd podczas przetwarzania pliku {audio_path}: {e}")