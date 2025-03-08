<h1 align="center">STAE:Sparse Temporal AutoEncoder for ECG Anomaly Detection</h1>

## Abstract
<image src="figures/tcnBlock.png">
  
Electrocardiogram (ECG) analysis is a fundamental tool for diagnosing various cardiac conditions. However, accurately distinguishing between normal and abnormal ECG signals remains a significant challenge due to high inter-individual variability and the inherent complexity of ECG waveforms. In this study, we propose a novel Sparse Temporal Autoencoder (STAE) for unsupervised ECG anomaly detection, leveraging Temporal Convolutional Networks (TCN) to extract deep hierarchical features from both time-domain and frequency-domain representations of ECG signals.
Unlike traditional approaches that require manually annotated abnormal ECG samples, our model is trained exclusively on normal ECG data, making it well-suited for real-world scenarios. STAE integrates TCN-based feature extraction with a masked signal reconstruction strategy, effectively capturing underlying temporal and spectral dependencies. Furthermore, STAE introduces a hybrid sparse attention mechanism, combining sparse block attention and sparse strided attention, which enhances the model's ability to focus on critical ECG patterns while maintaining computational efficiency.
The anomaly detection process is based on reconstruction errors, enabling robust differentiation between normal and abnormal signals. We evaluate STAE on the PTB-XL dataset, a large-scale benchmark for ECG analysis. Experimental results demonstrate that STAE achieves state-of-the-art performance in ECG anomaly detection, highlighting its potential as a powerful tool for automated and intelligent ECG analysis.

 

## Required Libraries
<ul>
  <li>Pytorch</li>
  <li>Numpy</li>
  <li>TQDM</li>
  <li>SciPy</li>
  <li>HeartPy</li>
  <li>PyWavelets</li>
  <li>Scikit-Learn</li>
</ul>


## Datasets
To assess the performance of our model, we conduct experiments using the PTB-XL dataset. The dataset preprocessing follows the approach outlined in <a href="https://github.com/UARK-AICV/TSRNet">TSRNet.</a> For further information, please refer to their repository.</a>

## Usage

### Training

```python
python train.py --data_path ./data/ --save_model 1 --save_path ./ckpt/STAE.pt
```

### Testing
```python
python test.py --data_path ./data/ --load_model 1 --load_path ./ckpt/STAE.pt
```

## Acknowledgment
A part of this code is adapted from these earlier works: [TSRNet.](https://github.com/UARK-AICV/TSRNet) and [Katharopoulos et al.](https://github.com/locuslab/TCN)

## Contact
If you have any questions, please feel free to create an issue on this repository or contact us at <radia.daci@isasi.cnr.it>.





