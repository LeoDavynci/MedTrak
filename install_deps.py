import sys
import subprocess

def install_dependencies():
    dependencies = [
        "mediapipe",
        "pyautogui",
        "opencv-python",
        "pycaw",
        "numpy"
    ]
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "--target=./libs"])
            print(f"{dep} installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {dep}: {e}")

if __name__ == "__main__":
    install_dependencies()
