## Overview

This repository contains reproduced vision experiments in  
_'A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks
'_, by Hendrycks et al.  
It uses Tensorflow Keras API to build model.

I tried to keep the code simple but still nasty enough to confuse you. Any suggestions or corrections are welcome.

### Abnormality module for anomaly detection

<img src="./images/abnormality_module.png" width="600px" align="center"/>  
The paper suggests utilizing **Anomality module** to enhance overall performance.

### Version range, Dependencies

-   python>=3.4
-   tensorflow>=1.8
-   numpy
-   scikit-learn(sklearn)
-   scikit-image(skimage)
-   h5py
-   jupyter(optional)

### Usage

```bash
python3 mnist_softmax.py
python3 mnist_auxiliary.py
```

### Todo:

-   Apply moving average to trained parameters, using tf.train.ExponentialMovingAverage()
-   Gelu Nonlinearity

### Resources:

-   [https://arxiv.org/abs/1610.02136](https://arxiv.org/abs/1610.02136)
-   [https://github.com/hendrycks/error-detection](https://github.com/hendrycks/error-detection) (Original repository)
-   [https://github.com/hendrycks/error-detection](https://github.com/hendrycks/error-detection) (Original repository)
