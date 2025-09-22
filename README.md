# ISIT 2025 Wireless Wanderlust Competition - Fading Fighter Team

## 🏆 Competition Results

- **Team Name:** Fading Fighter
- **Team Member:** Jafar Banar
- **Result:** **Third Prize** 🥉
- **Competition:** IEEE ISIT 2025 Wireless Wanderlust (D2I)
- **Conference:** IEEE International Symposium on Information Theory 2025

## 🚀 Key Innovation

**Intelligent Grant-Free Transmission for CSI-Based Localization: Achieving 70% Bandwidth Reduction with Competitive Accuracy**

Our approach introduced a novel dual-model ensemble architecture that addresses the fundamental trade-off between transmission efficiency and localization precision in CSI-based wireless systems.

## 📊 Results Summary

| Task | MAE (m) | R90 (m) | Transmission Rate | Key Achievement |
|------|---------|---------|------------------|-----------------|
| Task 1 | 0.4320 | 0.8841 | 100% | Baseline CSI localization |
| Task 2 | 0.4721 | 1.0018 | 100% | Temporal modeling |
| Task 3 | 0.3237 | 0.8016 | 30.02% | **70% bandwidth reduction** |
| Task 4 | 0.2517 | 0.6750 | 25% | **75% bandwidth reduction** |

## 🔬 Technical Approach

### Architecture Overview
- **Task 1:** 3D CNN with self-attention for spatial CSI feature extraction
- **Task 2:** CNN-LSTM hybrid leveraging temporal correlations across consecutive CSI measurements
- **Task 3:** Dual-model ensemble achieving **70% transmission reduction** with competitive accuracy
- **Task 4:** Multi-task learning achieving **75% transmission reduction** with best accuracy

### Key Technical Contributions
1. **Novel Architecture:** First integration of grant-free transmission principles with CSI-based localization
2. **Efficiency Gains:** Demonstrated 70-75% bandwidth reduction without sacrificing accuracy
3. **Robustness:** Performance maintained under adverse channel conditions (tested with 10% CSI dropout)
4. **Scalability:** Effective performance with varying numbers of users

## 📁 Repository Structure

```
competition-results/
├── README.md                    # This file
├── figures/                     # IEEE-compliant result figures
│   ├── mae_perfect.png         # MAE Performance Comparison
│   ├── r90_perfect.png         # R90 Performance Comparison
│   ├── transmission_clean_no_red.png  # Transmission Efficiency Analysis
│   └── combined_perfect.png    # Overall Performance Evaluation
├── scripts/                     # Figure generation scripts
│   └── create_perfectly_matched_figures.py
└── src/                        # Model implementation code
    ├── models/                 # Model architectures
    └── utils/                  # Utility functions
```

## 🛠️ Setup and Usage

### Prerequisites
```bash
pip install tensorflow matplotlib numpy seaborn pathlib
```

### Generate Figures
```bash
cd scripts/
python create_perfectly_matched_figures.py
```

### Run Models
```bash
cd src/
python -m models.basic_localization
```

## 📈 Performance Highlights

- **Best Accuracy:** Task 4 with MAE of 0.2517m
- **Best Efficiency:** 75% transmission reduction (Task 4)
- **Robust Performance:** Maintained accuracy under 10% CSI dropout
- **Competitive Results:** Third place in international competition

## 🎯 Competition Recognition

The work was recognized with **Third Prize** in the IEEE ISIT 2025 Wireless Wanderlust competition and was invited for presentation at the ISIT 2025 D2I Solution Showcase, validating the practical significance and competitive advantage of the proposed methodology.

## 📚 Technical Details

### Model Specifications
- **Architecture:** Dual-model ensemble (local + central models)
- **Input:** CSI measurements (4×8×16 complex tensors)
- **Output:** 2D position coordinates (x, y)
- **Training:** Semi-supervised learning with labeled/unlabeled data
- **Evaluation:** MEDE + R90 metrics

### Dataset
- **Source:** DICHASUS measurement collection
- **Size:** 25% labeled, 75% unlabeled samples
- **Precision:** Centimeter-level ground truth positioning
- **Environment:** Indoor localization scenarios

## 🤝 Contributing

This repository contains the complete implementation and results from the ISIT 2025 Wireless Wanderlust competition. Feel free to explore the code and build upon the methodology.

## 📄 License

This project is licensed under the MIT License - see the main repository LICENSE file for details.

---

*This repository showcases the technical implementation and results from the IEEE ISIT 2025 Wireless Wanderlust competition, demonstrating innovative approaches to CSI-based localization with grant-free transmission optimization.*
