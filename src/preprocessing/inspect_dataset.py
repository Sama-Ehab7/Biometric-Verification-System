
from pathlib import Path
from collections import Counter
from PIL import Image
import wave

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "vidtimit"
)


def inspect_video(speaker_path):

    video_path = speaker_path / "video"

    print("\n========== VIDEO INSPECTION ==========")

    image_files = [
        f for f in video_path.rglob("*")
        if f.is_file()
    ]

    print(f"Total files: {len(image_files)}")

    valid_images = 0
    invalid_images = 0

    dimensions = Counter()
    sequences = Counter()

    for file in image_files:

        try:
            with Image.open(file) as img:

                img.verify()

            with Image.open(file) as img:

                dimensions[img.size] += 1

            valid_images += 1
            sequences[file.parent.name] += 1

        except Exception as error:

            invalid_images += 1

            if invalid_images <= 5:
                print(f"Invalid: {file} | {error}")

    print(f"\nValid images: {valid_images}")
    print(f"Invalid images: {invalid_images}")

    print("\nImage Dimensions:")

    for size, count in dimensions.items():
        print(f"{size}: {count}")

    print("\nSequences:")

    for name, count in sorted(sequences.items()):
        print(f"{name}: {count} frames")


def inspect_audio(speaker_path):

    audio_path = speaker_path / "audio"

    print("\n========== AUDIO INSPECTION ==========")

    audio_files = sorted(audio_path.glob("*.wav"))

    print(f"Total WAV files: {len(audio_files)}")

    for file in audio_files:

        try:

            with wave.open(str(file), "rb") as audio:

                sample_rate = audio.getframerate()
                frames = audio.getnframes()

                duration = frames / sample_rate

                print(f"\nFile: {file.name}")
                print(f"Channels: {audio.getnchannels()}")
                print(f"Sample Rate: {sample_rate} Hz")
                print(f"Duration: {duration:.2f} seconds")
                print(f"Bit Depth: {audio.getsampwidth() * 8}")

        except Exception as error:

            print(f"Error: {file.name} | {error}")


def main():

    print("===== VidTIMIT Dataset Validation =====")

    speakers = [
        folder for folder in DATASET_PATH.iterdir()
        if folder.is_dir()
    ]

    print(f"Number of speakers: {len(speakers)}")

    for speaker in speakers:

        print(f"\nSpeaker: {speaker.name}")

        inspect_video(speaker)
        inspect_audio(speaker)


if __name__ == "__main__":
    main()
