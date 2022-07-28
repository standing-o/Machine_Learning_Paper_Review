## Batch normalization: Accelerating deep network training by reducing internal covariate shift.
- Authors : Ioffe, Sergey and Szegedy, Christian
- Journal : PMLR
- Year : 2015
- Link : https://arxiv.org/pdf/1502.03167.pdf

### Abstract
- Training Deep Neural Networks is complicated by the fact that the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change.
- This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities Internal covariate shift.
➔ Internal covariate shift
- `Batch normalization` : normalizing layer inputs for each training mini-batch

### Introduction
#### SGD Optimizer
- The gradient of the loss over a mini-batch
    <img src='https://user-images.githubusercontent.com/57218700/145709161-b3ef1dbd-5a17-4e75-b9ad-8c98462cbdc8.png' width=30%>
- It requires careful tuning of the model hyper-parameters, specifically the learning rate and the initial parameter values.
  
- Sigmoid activation function z = g(Wu + b). As |x| increases, g‘(x) tends to zero. (non-linearity saturation regime)
➔ The gradient flowing down to u will vanish and the model will train slowly
- `Batch normalization` : the distribution of nonlinearity inputs remains more stable as the network trains, then the optimizer would be less likely to get stuck in the saturated regime, and the training would accelerate.

### Internal Covariate Shift
- The change in the distributions of internal nodes of a deep network, in the course of training.
- `Batch normalization` takes a step towards reducing internal covariate shift.
➔ Accelerating the training of deep neural nets
➔ Reducing the dependence of gradients on the scale of the parameters or of their initial values i.e. allowing us to use much higher learning rates
➔ Regularizing the model and reducing the need for Dropout
➔ Using saturating nonlinearities by preventing the network from getting stuck in the saturated modes

### Normalization via mini-batch statistics
1. Normalize each scalar feature independently, by making it have zero mean and unit variance.
- Normalize each dimension where the expectation and variance are computed over the training data set.
    <img src='https://user-images.githubusercontent.com/57218700/145709478-d90441e8-aa9e-4ddc-a447-0395dcecc859.png' width=30%>
- Normalizing the inputs of a sigmoid would constrain them to the linear regime of the nonlinearity.
  - For each activation 𝑥<sup>(k)</sup>, a pair of trainable parameters γ<sup>(k)</sup>, β<sup>(k)</sup>, which scale and shift the normalized value : y<sup>(k)</sup> = &gamma;<sup>(k)</sup>x&#770;<sup>(k)</sup> + &beta;<sup>(k)</sup>.
  - Recover the original activations, if that were the optimal thing to do.
2. Use mini-batches in stochastic gradient training, each mini-batch produces estimates of the mean and variance of each activation.
    <src img='https://user-images.githubusercontent.com/57218700/145709825-22e5fa47-517a-44d3-9c0d-6cea5618b2ab.png' width=50%>

### Training and Inference with `Batch normalization` networks
- The normalization of activations that depends on the mini-batch allows efficient training, but is neither necessary nor desirable during inference.
- We want the output to depend only on the input, deterministically. For this, once the network has been trained, we use the normalization.
  ➔ Since the means and variances are fixed during inference, the normalization is simply a linear transform applied to each activation.
    <img src='https://user-images.githubusercontent.com/57218700/145709922-8f6cf499-2c3c-4ce4-bf2f-76811e69b6d8.png' width=50%>

### Batch-normalized convolutional networks
- We add the `BN` transform immediately before the nonlinearity by normalizing x = Wu + b.

### `Batch Normalization` enables higher learning rates
- `Batch Normalization` makes training more resilient to the parameter scale.
- Back-propagation through a layer is unaffected by the scale of its parameter, so larger weights lead to smaller gradients, and `batch normalization` will stabilize the parameter growth.
    <img src='https://user-images.githubusercontent.com/57218700/145710086-e057f54b-e8d1-4f6c-9180-91c4b8d834cb.png' width=30%>

### Experiment
- `Batch normalization` makes the distribution more stable and reduces the internal covariate shift.
- Test accuracy on MNIST
    <img src='https://user-images.githubusercontent.com/57218700/145710133-43b9d40a-246c-48ae-857e-0e99d204b0b2.png' width=70%>