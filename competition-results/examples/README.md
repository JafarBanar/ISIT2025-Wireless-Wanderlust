# CSI-Based Localization Examples

This directory contains practical examples demonstrating how to use the grant-free CSI-based localization system that achieved **3rd place** in the IEEE ISIT 2025 Wireless Wanderlust competition.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Quick Demo (Recommended First Step)

```bash
python demo.py
```

## 📁 Files Overview

### `demo.py`
Quick demonstration showcasing:
- **70% bandwidth reduction** achievement
- **Dual-model architecture** visualization
- **Performance metrics** from competition
- **Key innovation** explanation

### `requirements.txt`
All necessary dependencies for running the examples.

## 🏗️ Architecture Overview

### Grant-Free Model Components

1. **Local Model**: Makes intelligent transmission decisions
   - 3D CNN for spatial CSI feature extraction
   - Binary classification for transmission probability
   - Achieves 70% bandwidth reduction

2. **Central Model**: Predicts position coordinates
   - 3D CNN with self-attention mechanism
   - Dense layers for coordinate regression
   - Maintains competitive accuracy (0.25m MAE)

3. **Ensemble Fusion**: Combines both outputs
   - Transmission-aware position prediction
   - Optimized for competition metrics

## 📊 Key Results

| Metric | Value | Achievement |
|--------|-------|-------------|
| **Bandwidth Reduction** | **70%** | 🎯 Main Innovation |
| **Localization Accuracy** | **0.25m MAE** | 🏆 Competitive Performance |
| **Transmission Rate** | **30%** | ⚡ Efficient Communication |
| **Competition Result** | **3rd Place** | 🥉 IEEE ISIT 2025 |

## 🎯 Competition Innovation

### Grant-Free Transmission
- **Problem**: Traditional CSI-based localization requires continuous transmission
- **Solution**: Intelligent transmission decisions based on CSI quality
- **Result**: 70% reduction in bandwidth usage

### Dual-Model Architecture
- **Local Model**: Decides when to transmit based on CSI characteristics
- **Central Model**: Predicts position when transmission is needed
- **Fusion**: Combines both decisions for optimal performance

## 📞 Contact

- **Email**: jaafar.banar@gmail.com
- **Institution**: Chalmers University of Technology
- **GitHub**: [@JafarBanar](https://github.com/JafarBanar)

---

*This implementation demonstrates the winning approach from the IEEE ISIT 2025 Wireless Wanderlust competition, showcasing innovative grant-free transmission for CSI-based localization.*
