import csv
import re
import time
from pathlib import Path

import jiwer
import whisper

AUDIO_EXTENSIONS = {".wav", ".mp3"}


def find_audio_files(root):
    return sorted(
        p for p in Path(root).rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def read_text(txt_path):
    try:
        return txt_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return txt_path.read_text(encoding="windows-1250").strip()


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def calculate_wer(reference, hypothesis):
    reference = normalize(reference)
    hypothesis = normalize(hypothesis)
    if not reference:
        return float("nan")
    if not hypothesis:
        return 1.0
    return jiwer.wer(reference, hypothesis)


def process_audio_files(base_directory, output_csv="wyniki_transkrypcji.csv", model_size="tiny"):
    print(f"Ładowanie modelu Whisper ({model_size})...")
    model = whisper.load_model(model_size, device="cuda")
    print(f"Model załadowany. Wyniki będą zapisywane do pliku: {output_csv}\n")

    audio_files = find_audio_files(base_directory)
    print(f"Znaleziono {len(audio_files)} plików audio\n")

    with open(output_csv, mode="w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "Folder", "Nazwa_pliku", "Czas_trwania_Tn_s",
            "Czas_przetwarzania_Tp_s", "RTF", "WER_procent",
        ])

        for audio_path in audio_files:
            txt_path = audio_path.with_suffix(".txt")

            if not txt_path.exists():
                print(f"Pominięto: {audio_path.name} (brak pliku transkrypcji)")
                continue

            reference_text = read_text(txt_path)
            if not reference_text:
                print(f"Pominięto: {audio_path.name} (pusty plik tekstowy)")
                continue

            try:
                audio = whisper.load_audio(str(audio_path))
                t_n = len(audio) / whisper.audio.SAMPLE_RATE

                start_time = time.time()
                result = model.transcribe(audio, language="pl")
                t_p = time.time() - start_time

                hypothesis_text = result["text"].strip()


                rtf = t_p / t_n
                wer_percentage = calculate_wer(reference_text, hypothesis_text) * 100

                # save to .csv
                writer.writerow([
                    audio_path.parent.name,
                    audio_path.name,
                    f"{t_n:.2f}",
                    f"{t_p:.2f}",
                    f"{rtf:.4f}",
                    f"{wer_percentage:.2f}",
                ])
                csv_file.flush()

                print(f"Przetworzono: {audio_path.parent.name}/{audio_path.name} "
                      f"| WER: {wer_percentage:.2f}% | RTF: {rtf:.4f}")

            except Exception as e:
                print(f"Błąd podczas przetwarzania {audio_path}: {e}")


if __name__ == "__main__":
    KATALOG_GLOWNY = "./recordings_stt"
    PLIK_WYNIKOWY = "wyniki_transkrypcji.csv"

    process_audio_files(KATALOG_GLOWNY, output_csv=PLIK_WYNIKOWY, model_size="tiny")