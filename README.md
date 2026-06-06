# 🥊 Boxing Arcade

An interactive, real-time arcade game built with Python that uses computer vision to track your physical movements instead of a traditional game controller! By leveraging **MediaPipe's Pose Landmarker**, the game analyzes shoulder-to-wrist extensions to detect when you throw a left hook or a right cross.

## 🚀 Features
* **Real-Time Human Pose Tracking:** Driven by MediaPipe vision models running inside an optimized OpenCV loop.
* **Dynamic Physics Engine:** Simulates realistic swinging animations for a heavy bag via smooth angular interpolation (LERP) upon punch impacts.
* **Smart UI Layout:** Rendered on a fixed, stable 1600x900 canvas layout to ensure text components and skeleton lines scale perfectly across different webcams.
* **Instant Keyboard Controls:** Built-in keyboard listener—press **'R'** to immediately reset the 30-second chronometer round and clear scores.

---

## 👷 Installation

Follow these quick steps to install the required libraries and run the game on your machine.

### Step 1: Clone the Repository

git clone https://github.com/raedbenbettaieb-cyber/AI-Boxing-Arcade-Game.git
cd AI-Boxing-Arcade-Game

### Step 2: Install Dependencies

pip install opencv-python mediapipe

### Step 3: Run the Game
python game.py
