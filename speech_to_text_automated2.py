import os
import time
import wave
import csv
import whisper
import jiwer
from pathlib import Path


def get_audio_duration(file_path):
    """Zwraca czas trwania nagrania w sekundach."""
    with wave.open(str(file_path), 'rb') as audio_file:
        frames = audio_file.getnframes()
        rate = audio_file.getframerate()
        duration = frames / float(rate)
        return duration


def calculate_wer(reference, hypothesis):
    """Oblicza Word Error Rate (WER)."""
    try:
        error_rate = jiwer.wer(reference, hypothesis)
        return error_rate
    except ValueError:
        return float('inf')


def process_audio_files(base_directory, output_csv="wyniki_transkrypcji.csv", model_size="base"):
    """Przetwarza pliki w folderach, oblicza WER/RTF i zapisuje do CSV."""
    print(f"Ładowanie modelu Whisper ({model_size})...")
    model = whisper.load_model(model_size)
    print(f"Model załadowany. Wyniki będą zapisywane do pliku: {output_csv}\n")

    base_path = Path(base_directory)

    # Otwarcie pliku CSV do zapisu
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Zapisanie nagłówków kolumn
        writer.writerow(['Nazwa_pliku', 'Czas_trwania_Tn_s', 'Czas_przetwarzania_Tp_s', 'RTF', 'WER_procent'])

        # Rekursywne wyszukiwanie wszystkich plików .wav
        for wav_path in base_path.rglob("*.wav"):
            txt_path = wav_path.with_suffix('.txt')

            if not txt_path.exists():
                print(f"Pominięto: {wav_path.name} (brak pliku transkrypcji)")
                continue

            with open(txt_path, 'r', encoding='utf-8') as f:
                reference_text = f.read().strip()

            if not reference_text:
                print(f"Pominięto: {wav_path.name} (pusty plik tekstowy)")
                continue

            # 1. Pobranie czasu trwania nagrania
            try:
                t_n = get_audio_duration(wav_path)
            except Exception as e:
                print(f"Błąd odczytu pliku audio {wav_path.name}: {e}")
                continue

            # 2. Transkrypcja i pomiar czasu
            start_time = time.time()
            result = model.transcribe(str(wav_path))
            end_time = time.time()

            t_p = end_time - start_time
            hypothesis_text = result["text"].strip()

            # 3. Obliczenia
            rtf = t_p / t_n
            wer = calculate_wer(reference_text, hypothesis_text)
            wer_percentage = wer * 100

            # 4. Zapis do pliku CSV
            writer.writerow([
                wav_path.name,
                f"{t_n:.2f}",
                f"{t_p:.2f}",
                f"{rtf:.4f}",
                f"{wer_percentage:.2f}"
            ])

            # Wyświetlenie postępu w konsoli
            print(f"Przetworzono: {wav_path.name} | WER: {wer_percentage:.2f}% | RTF: {rtf:.4f}")


if __name__ == "__main__":
    KATALOG_GLOWNY = "./recordings_stt"
    PLIK_WYNIKOWY = "wyniki_transkrypcji.csv"

    process_audio_files(KATALOG_GLOWNY, output_csv=PLIK_WYNIKOWY, model_size="base")