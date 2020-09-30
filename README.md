# Digital Speech Signal Processing - Course Exercises

This repository contains the complete set of exercises and supplementary materials for the Digital Speech Signal Processing course conducted during the Summer Semester 2020 at the University of Paderborn as part of the M.Sc. program.

## Course Overview

This course covers fundamental and advanced topics in digital speech signal processing, progressing from basic signal analysis to sophisticated audio processing techniques.

## Exercise Topics

### Core Exercises

**Ex01: Continuous and Discrete Time Fourier Transform**
- Signal representation in time and frequency domains
- Fourier analysis fundamentals

**Ex02: Sampling and Reconstruction**
- Nyquist sampling theorem
- Signal reconstruction from samples
- Quantization effects
- Aliasing phenomena

**Ex03: Energy, Discretization and Parseval's Theorem**
- Signal energy and power calculations
- Parseval's theorem applications
- Discretization effects

**Ex04: DFT and Computational Effort**
- Discrete Fourier Transform implementation
- Computational complexity analysis
- FFT algorithm efficiency

**Ex05: Partitioned Block Overlap Save**
- Convolution techniques for long signals
- Block processing methods
- Overlap-save algorithm implementation

**Ex06: Window Functions and Leakage Effect**
- Spectral leakage analysis
- Window function characteristics
- Frequency resolution trade-offs

**Ex07: Periodograms and ACF Estimation and MFCC**
- Power spectral density estimation
- Autocorrelation function analysis
- Mel-frequency cepstral coefficients
- Feature extraction for speech processing

**Ex08: Linear Estimator for Auto-regressive Processes (with LPC)**
- Linear prediction coding
- Auto-regressive modeling
- Speech analysis applications

**Ex09: Noise Reduction**
- Spectral subtraction methods
- Wiener filtering
- Speech enhancement techniques

**Ex11: Adaptive Filters**
- LMS and NLMS algorithms
- Adaptive noise cancellation
- System identification applications

### Supplementary Materials

**Supplement 1: Fourier Series**
- Fourier series decomposition
- Harmonic analysis
- Signal synthesis

**Supplement 2: From Fourier Series to FFT**
- Algorithm evolution
- Computational optimization
- FFT implementation details

**Supplement 3: Short Term Fourier Transform**
- Time-frequency analysis
- Spectrogram representation
- STFT windowing strategies

## Topic Relationships and Progression

The exercises follow a logical progression building upon fundamental concepts:

```
Ex01 (Fourier Transform) → Ex02 (Sampling) → Ex03 (Energy & Parseval) → Ex04 (DFT/FFT)
                                   ↓
                           Ex06 (Window Functions) → Ex05 (Block Convolution)
                                   ↓
                           Ex07 (Spectral Analysis) → Ex08 (Linear Prediction)
                                   ↓
                           Ex09 (Noise Reduction) → Ex11 (Adaptive Filters)
```

Supplementary materials provide theoretical foundations:
```
Supplement 1 (Fourier Series) → Supplement 2 (FFT Development) → Supplement 3 (STFT)
```

## Technical Implementation

The exercises demonstrate various signal processing techniques using:
- Python implementations with NumPy and SciPy
- MATLAB scripts for algorithm development
- Jupyter notebooks for interactive analysis
- Real audio signal examples

## Course Context

- **Institution**: University of Paderborn
- **Program**: M.Sc. (Master of Science)
- **Semester**: Summer Semester 2020 (April - September)
- **Duration**: 6 months of progressive learning

## File Structure

Each exercise directory contains:
- Template files for initial implementation
- Solution files with complete implementations
- Scratch files for development and experimentation
- Supporting audio files for real-world testing
- Comprehensive documentation and analysis

## Prerequisites

Basic understanding of:
- Signal and systems theory
- Linear algebra
- Probability theory
- Programming fundamentals (Python/MATLAB)

## Learning Outcomes

Upon completion, students will have practical experience with:
- Digital signal processing fundamentals
- Speech analysis and synthesis techniques
- Audio enhancement and noise reduction
- Adaptive filtering applications
- Feature extraction for speech processing
