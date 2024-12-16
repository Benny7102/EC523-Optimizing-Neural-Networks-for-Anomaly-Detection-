# EC523 - Anomaly Detection Optimization

This project focuses on optimizing the BatchNorm-based Weakly Supervised Video Anomaly Detection (BN-WVAD) model for real-time crime detection in video footage, with an emphasis on improving computational efficiency while maintaining detection performance.

## Project Overview

We aim to enhance the BN-WVAD model through two main optimization approaches:
- **Pruning**: Implementing the lottery ticket hypothesis to identify efficient subnetworks
- **Quantization**: Applying post-training quantization techniques to reduce model complexity

### Dataset
- UCF-Crime Dataset
- 1,900 untrimmed surveillance videos
- 1,610 training videos and 290 test videos
- Contains both normal and anomalous events

### Performance Metrics
- AUC (Area Under Curve)
- AP (Average Precision)
- AUC_sub/AP_sub for known anomalies
- Inference time (ms/frame or FPS)

## Getting Started

[Installation instructions to be added]

## Project Timeline

| Task | Deadline |
|------|----------|
| Data Preparation and Preprocessing | 10/18/24 |
| Baseline Model Implementation | 10/25/24 |
| Lottery Ticket Implementation | 11/01/24 |
| Model Optimization and Tuning | 11/08/24 |
| Performance Analysis and Comparison | 11/08/24 |
| Status Report | 11/13/24 |
| Final Optimizations and Documentation | 11/27/24 |
| Final Project Report and Presentation | 12/01/24 |

## Team

- **Pruning Implementation**
  - Jimmy (Jialin Sui)
  - Jason Li

- **Quantization Implementation**
  - Benny Li
  - Nathan Lee
  - Juan De Carcer

## References

1. Zhou, Y., et al. (2023). BatchNorm-based Weakly Supervised Video Anomaly Detection. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*.

2. Frankle, J., & Carbin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. *International Conference on Learning Representations (ICLR)*.

3. Sultani, W., Chen, C., & Shah, M. (2018). Real-world anomaly detection in surveillance videos. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 6479–6488.

4. Chen, J., et al. (2021). Quantization-aware Optimization Approach for CNNs Inference on CPUs. *arXiv preprint arXiv:2103.13630*.

5. Liu, Y., Chen, T., & Xiong, Z. (2021). Optimizing CNN Model Inference on CPUs. *arXiv preprint arXiv:2106.08295*.

## License

[License information to be added]

## Contributing

[Contribution guidelines to be added]
