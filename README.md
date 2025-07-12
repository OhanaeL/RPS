# Rock Paper Scissors - MediaPipe

A real-time Rock Paper Scissors game that uses MediaPipe for hand gesture recognition, combined with face filters and score tracking.

---

## Features

- Real-time hand gesture detection with MediaPipe Hands.
- Face filters for win/lose effects using MediaPipe Face Mesh.
- Scoreboard and round tracking.
- User-friendly controls.
- Supports multiple cameras (webcam and USB cameras).

---

## Requirements

- Python 3.7 or higher
- OpenCV
- MediaPipe
- NumPy

All required Python packages are listed in `requirements.txt`.

---

## Setup and Installation

### Windows (PowerShell)

1. Open PowerShell and navigate to the project directory:

```powershell
cd "D:\Github Repositories\RPS"
```

Create a virtual environment (if you haven't already):

```powershell
python -m venv venv
```

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

Run the game:

```powershell
python RPS.py
```
