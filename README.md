# AFT – Autonomous Freight Truck 🚛

AFT is an intelligent self-driving truck system developed using **Euro Truck Simulator 2 (ETS2)** and **CARLA**, implementing and comparing multiple approaches to autonomous driving — including **Computer Vision**, **Deep Learning**, and **Reinforcement Learning**.

---

## 📅 Project Duration

**March 2025 – May 2025**

---

## 📌 Summary

This project aims to explore and implement autonomous driving logic tailored for heavy-duty freight trucks. Using **Euro Truck Simulator 2** as a real-time simulation platform and **CARLA** for reinforcement learning environments, the system compares three major approaches:

- **Computer Vision–based lane detection** using traditional techniques  
- **End-to-End Deep Learning** using CNNs trained on driving data  
- **Reinforcement Learning (DQN)** using the CARLA simulator

---

## 🚛 Simulation Platforms Used

### Euro Truck Simulator 2 (ETS2)
Used as the main testbed for implementing real-time autonomous truck control:
- High-fidelity road simulation
- Realistic truck physics
- Real-time telemetry access via **ETS2 Telemetry SDK**

### CARLA Simulator
Used for reinforcement learning:
- Supports custom agents and reward environments
- Open-source and flexible for deep RL tasks

---

## 🧠 Techniques Implemented

### 1. Computer Vision (CV)
- **Canny Edge Detection**
- **Hough Line Transform**
- Lane region masking and ROI isolation
- Real-time steering angle estimation

### 2. Deep Learning (DL)
- End-to-End **Convolutional Neural Network (CNN)**
- Regression-based output for steering angle
- Trained on video frames + steering logs from ETS2

### 3. Reinforcement Learning (RL)
- **DQN (Deep Q-Network)** agent in CARLA
- Environment rewards for staying in lane, avoiding collisions, and efficient driving

---

## 📂 Repository Structure

```
/
├── data/               # Collected driving footage and labels from ETS2
├── carla_rl/           # DQN implementation using CARLA simulator
├── vision/             # Classical CV lane detection scripts
├── models/             # CNN models for ETS2 (train/test)
├── utils/              # Supporting scripts for steering, preprocessing
├── gta/                # Sentdex-based GTA-V code (initial learning only)
├── requirements.txt
└── README.md
```

---

## ⚙️ Getting Started

### 🔧 Prerequisites

Install dependencies using:
```bash
pip install -r requirements.txt
```

Additional requirements:
- **ETS2 with Telemetry SDK**: [ETS2 Telemetry Plugin](https://github.com/nlhans/ets2-sdk-plugin)
- **CARLA simulator**: [carla.org](https://carla.org)

---

### ▶️ Running the Project

#### 🔹 1. Vision-based Lane Detection
```bash
python vision/lane_detection.py
```

#### 🔹 2. Deep Learning Model Training
```bash
python models/train_cnn_model.py
```

#### 🔹 3. Deep Learning Inference (Real-Time in ETS2)
```bash
python models/test_cnn_model.py
```

#### 🔹 4. Reinforcement Learning in CARLA
Make sure CARLA server is running:
```bash
./CarlaUE4.sh  # or CarlaUE4.exe
python carla_rl/train_dqn.py
python carla_rl/evaluate_agent.py
```

---

## 📊 Results and Observations

| Approach           | Simulator Used | Strengths                             | Limitations                        |
|--------------------|----------------|----------------------------------------|------------------------------------|
| Computer Vision    | ETS2           | Lightweight, interpretable             | Sensitive to lighting/curves       |
| Deep Learning (CNN)| ETS2           | Handles nonlinear steering well        | Requires large labeled dataset     |
| Reinforcement RL   | CARLA          | Learns from interaction, generalizable | Long training time, sparse rewards |

---

## 🔗 Useful References

- 🧰 ETS2 Telemetry SDK: https://github.com/nlhans/ets2-sdk-plugin  
- 📦 CARLA Simulator: https://carla.org/  
- 📘 NVIDIA PilotNet Paper: https://developer.nvidia.com/blog/deep-learning-self-driving-cars/  
- 📜 OpenCV Hough Transform: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html  
- 📙 Reinforcement Learning with DQN: https://www.tensorflow.org/agents/tutorials/0_intro_rl  

---

## 🧩 Learning Inspirations

- **Sentdex's GTA‑V Deep Learning Tutorial**: Used as an early prototype reference for capturing screen data and applying CNN-based models. Code in the `gta/` folder is based on his work and was adapted during the initial learning phase.  
  - GitHub: https://github.com/Sentdex/pygta5  
  - YouTube Series: https://www.youtube.com/playlist?list=PLQVvvaa0QuDfSfqQuee6K8opKtZsh7sA9

---

## 🧪 Future Enhancements

- Add **object detection (YOLO)** for detecting vehicles and pedestrians  
- Improve **reward shaping** in reinforcement learning for smoother driving  
- Simulate **multi-camera setups** with front and rear views  
- Integrate with **ROS** for potential real-world hardware testing

---

## 📎 Project Citation (for Resume)

**Autonomous Driving System for Trucks | Python, CARLA, TensorFlow, PyTorch, Git**  
📅 March 2025 – May 2025  
• Developed and compared three autonomous driving approaches—Reinforcement Learning (DQN in CARLA), Computer Vision, and Deep Learning in Euro Truck Simulator 2.  
• Implemented real-time lane detection and steering control using Canny Edge Detection, Hough Transform, and CNNs.  
• Integrated control system with live Euro Truck Simulator 2 gameplay.  
• GitHub: [github.com/hevardhan/AFT](https://github.com/hevardhan/AFT)

---

## 🪪 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
