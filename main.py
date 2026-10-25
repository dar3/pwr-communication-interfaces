import whisper
import queue
import sys
import sounddevice as sd
import numpy as np

# Audio settings
SAMPLE_RATE = 16000  # Whisper expects a 16kHz sample rate
DURATION = 3  # Transcribe in 3-second blocks
BLOCK_SIZE = SAMPLE_RATE * DURATION

# Queue to hold audio chunks
audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    """This function is called for each audio block by sounddevice."""
    if status:
        print(status, file=sys.stderr)
    # Store a copy of the recorded audio data into the queue
    audio_queue.put(indata.copy())


def main():
    print("Loading Whisper model (this may take a moment)...")
    # You can change "base" to "tiny", "small", "medium", or "large" depending on your hardware
    model = whisper.load_model("base")
    print("Model loaded. Listening... (Press Ctrl+C to stop)")

    # Open the microphone stream
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                            callback=audio_callback, blocksize=BLOCK_SIZE):
            while True:
                # Get the next chunk of audio from the queue
                audio_chunk = audio_queue.get()

                # Flatten the 2D array to a 1D array as expected by Whisper
                audio_data = audio_chunk.flatten()

                # Transcribe the audio chunk
                # fp16=False prevents warnings if you are running on a CPU
                result = model.transcribe(audio_data, fp16=False)

                text = result['text'].strip()
                if text:
                    print(f"You said: {text}")

    except KeyboardInterrupt:
        print("\nStopping transcription.")


if __name__ == "__main__":
    main()