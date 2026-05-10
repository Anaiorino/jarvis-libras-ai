import subprocess


def speak(text):
    text = text.replace("'", "")

    subprocess.Popen(
        [
            "powershell",
            "-Command",
            f"Add-Type -AssemblyName System.Speech; "
            f"$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speak.Rate = 0; "
            f"$speak.Volume = 100; "
            f"$speak.Speak('{text}');"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )