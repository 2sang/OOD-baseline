## Overview

This repository contains reproduced experiments in  
_'A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks
'_, by Hendrycks et al.  
it uses Tensorflow Keras API.

I tried to keep the code simple but still nasty enough to confuse you. Any suggestions or corrections are welcome.

### Usage

```bash
python3 mnist_softmax.py
```

### Version range, Dependencies

-   python>=3.4
-   numpy
-   tensorflow>=1.8
-   sklearn
-   jupyter(optional)

### Todo:

-   Apply moving average to trained parameters, using tf.train.ExponentialMovingAverage()
-   Gelu Nonlinearity

### Resources:

-   [https://arxiv.org/abs/1610.02136](https://arxiv.org/abs/1610.02136)
-   [https://github.com/hendrycks/error-detection](https://github.com/hendrycks/error-detection) (Original repository)
