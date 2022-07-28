## Adam: A method for stochastic optimization
- Authors : Kingma, Diederik P and Ba, Jimmy
- Journal : arXiv preprint arXiv
- Year : 2014
- Link : https://arxiv.org/pdf/1412.6980.pdf

### Abstract
- `Adam` is...
➔ Algorithm for first-order gradient-based optimization of stochastic objective functions based on adaptive estimates of lower-order moments.
➔ Straightforward to implement, computationally efficient, little memory requirements, invariant to diagonal rescaling of the gradients, well suited for problems that are large in terms of data or params.
➔ Appropriate for non-stationary objectives and problems with very noisy and sparse gradients
➔ Its hyper params have intuitive interpretations and typically require little tuning.

### Introduction
- If an objective function requiring maximization or minimization with respect to its parameters is differentiable, gradient descent is a relatively efficient optimization method, since the computation of first-order partial derivatives is of the same computational complexity as just evaluating the function.
- `Adam` computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.
➔ designed to combine the advantages of AdaGrad and RMSProp.
➔ **Advantages** : magnitudes of param updates are invariant to rescaling of the gradient, its stepsizes are approximately bounded by the stepsize hyperparameters.

### `Adam` algorithm pseudo-code
<img src="https://user-images.githubusercontent.com/57218700/145706141-1c4ce028-4770-4c41-ba1e-bea61f79d32a.png" width="75%">

### `Adam` algorithm
- The algorithm updates exponential moving averages of the gradient (m) and The squared gradient (v) where the hyper-params &beta;<sub>1</sub>, &beta;<sub>2</sub> control the exponential decay rates of these moving averages.
- These moving averages are initialized as 0, leading to moment estimates that are biased towards zero, especially during the initial timesteps, and especially when the decay rates are small (i.e &beta; are close to 1)
    <img src="https://user-images.githubusercontent.com/57218700/145706524-f24556e2-1a00-43ba-a774-fae382017120.png" width="31.5%"> <img src="https://user-images.githubusercontent.com/57218700/145706532-bd8dfdfa-569c-4105-b2cd-6717b747de2d.png" width="30%">

### Initialization bias correction
-  Let us initialize the exponential moving average as v<sub>0</sub> = 0, then v<sub>t</sub> can be written as a function of the gradients at all previous timesteps:
    <img src="https://user-images.githubusercontent.com/57218700/145706646-90c1640b-8289-4389-9f74-c1421da04c30.png" width="30%">
    <img src="https://user-images.githubusercontent.com/57218700/145706679-cf0b26a7-c89c-4b2e-a110-f77671a51752.png" width="40%">
    where ζ = 0 if the true second moment E[g<sub>i</sub><sup>2</sup>] is stationary (can be kept small).
- we divide by (1-&beta;<sub>2</sub><sup>t</sup> to correct the initialization bias.

### Experiment
- Logistic regression training negative log likelihood on MNIST images
    <img src="https://user-images.githubusercontent.com/57218700/145706857-3042a1f4-2c3d-427c-a3a4-de8d31f9460e.png" width="30%">
- Training of multilayer neural networks on MNIST images
    <img src="https://user-images.githubusercontent.com/57218700/145706877-967e507d-8ee5-4ccc-9954-b6e558cb0594.png" width="30%">
- Convolutional neural networks training cost on CIFAR10
    <img src="https://user-images.githubusercontent.com/57218700/145706902-0397d29c-5aa5-4f75-a62e-40df8b680147.png" width="70%">