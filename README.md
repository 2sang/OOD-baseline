## Overview

This repository contains reproduced experiments in  
_'A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks
'_, by Hendrycks et al.  
It uses Tensorflow Keras API to build model.

I tried to keep the code simple but still nasty enough to confuse you. Any suggestions or corrections are welcome.

### Version range, Dependencies

-   python>=3.4
-   tensorflow>=1.8
-   numpy
-   sklearn
-   jupyter(optional)

### Usage

```bash
python3 mnist_softmax.py
```

### Todo:

-   Apply moving average to trained parameters, using tf.train.ExponentialMovingAverage()
-   Gelu Nonlinearity

### Resources:

-   [https://arxiv.org/abs/1610.02136](https://arxiv.org/abs/1610.02136)
-   [https://github.com/hendrycks/error-detection](https://github.com/hendrycks/error-detection) (Original repository)
