
"""Capture image on the pi, which will be ssh'd into, then send it back and display it on the computer"""

from pathlib import Path
import subprocess

import cv2
import numpy as np


OUTPUT_PATH = Path(__file__).with_name("camera_test.jpg")
REMOTE_USER = "hjohns"
REMOTE_HOST = "192.168.1.117"
REMOTE_PATH = "/home/hjohns/Downloads/camera_test_{}.jpg".format(np.random.randint(0, 1000))


def run_remote_command(
    command: list[str],
    description: str,
    *,
    required: bool = True,
) -> bool:
    """Run a remote command and print enough output to diagnose failures."""
    print(f"[debug] {description}: {' '.join(command)}", flush=True)
    result = subprocess.run(command, capture_output=True, text=True)

    if result.stdout:
        print(f"[debug] stdout:\n{result.stdout.rstrip()}", flush=True)
    if result.stderr:
        print(f"[debug] stderr:\n{result.stderr.rstrip()}", flush=True)
    print(f"[debug] exit code: {result.returncode}", flush=True)

    if result.returncode != 0 and required:
        raise RuntimeError(
            f"{description} failed with exit code {result.returncode}"
        )
    return result.returncode == 0


# --------------------------------------------------------------------------
# Image capture
# --------------------------------------------------------------------------

def capture_image() -> np.ndarray:
    capture_args = ['rpicam-jpeg', '-o', '-', '--width', '3280', '--height', '2464', '--nopreview']
    result = subprocess.run(capture_args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    image = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # immediately convert to grayscale to speed up processing
    return gray

def main() -> None:
    """Capture an image, save it locally, and display it."""
    image = capture_image()

    if not cv2.imwrite(str(OUTPUT_PATH), image):
        raise RuntimeError(f"Failed to save captured image to {OUTPUT_PATH}")

    print(f"[debug] Captured image saved to {OUTPUT_PATH}", flush=True)

    remote = f"{REMOTE_USER}@{REMOTE_HOST}"
    opened = run_remote_command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            remote,
            "printf 'Connected as '; whoami; printf ' on '; hostname",
        ],
        "SSH connectivity check",
    )
    run_remote_command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            remote,
            "test -d /home/hjohns/Downloads",
        ],
        "Remote destination check",
    )
    run_remote_command(
        [
            "scp",
            str(OUTPUT_PATH),
            f"{remote}:{REMOTE_PATH}",
        ],
        "Image transfer",
    )
    run_remote_command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            remote,
            "test -f " + REMOTE_PATH,
        ],
        "Remote file check",
    )

    run_remote_command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            remote,
            "xdg-open",
            REMOTE_PATH,
        ],
        "Remote image open",
        required=False,
    )
    if opened:
        print("[debug] Image opened on the computer", flush=True)
    else:
        print(
            "[debug] Image transferred successfully, but the remote host "
            "could not open it automatically",
            flush=True,
        )


if __name__ == "__main__":
    main()
