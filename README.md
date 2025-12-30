# Virtual Mouse

Control your computer's mouse cursor using hand gestures through your webcam.

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- MediaPipe
- PyAutoGUI
- NumPy
- pynput

## Installation

```bash
pip install opencv-python mediapipe pyautogui numpy pynput
```

## Usage

```bash
python main.py
```

Press `q` to quit.

## Hand Gestures

| Gesture | Action |
|---------|--------|
| **Pinch** (thumb + index close, index straight) | Move cursor |
| **Index bent** + middle straight + thumb away | Left click |
| **Middle bent** + index straight + thumb away | Right click |
| **Both bent** + thumb away | Double click |

## Tips

- Ensure good lighting for best hand detection
- Keep your hand within camera view
- DPI multiplier is set to 2.5x for enhanced sensitivity
- Adjust `dpi_multiplier` in code if cursor moves too fast/slow
