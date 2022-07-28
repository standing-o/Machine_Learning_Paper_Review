## Delving deep into rectifiers: Surpassing human-level performance on imagenet classification
- Authors : He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian
- Journal : IEEE
- Year : 2015
- Link : https://arxiv.org/pdf/1502.01852.pdf

### Abstract
- We propose a **Parametric Rectified Linear Unit (`PReLU`)** that generalizes the traditional rectified unit.
➔ improves model fitting with nearly zero extra computational cost and little overfitting risk.
- We derive a robust initialization method that particularly considers the rectifier nonlinearities.
➔ enables us to train extremely deep rectified models directly from scratch and to investigate deeper or wider network architectures.
- Our result is the first to surpass the reported human-level performance (5.1%) on ImageNet.

### Introduction
- ReLU expedites convergence of the training procedure and leads to better solutions than conventional sigmoid like units.
  - ReLU is not a symmetric function, the mean response of ReLU is always no smaller than zero.
- By explicitly modeling the nonlinearity of rectifiers (ReLU/`PReLU`), we derive a theoretically sound initialization method, which helps with convergence of very deep models.
- Leaky ReLU (a=0.01)
    <img src='https://user-images.githubusercontent.com/57218700/145715026-79d559ff-c8ad-4332-af49-71842a4cda32.png' width=50%>

### Parametric Rectifiers
- `PReLU` introduces a very small number of extra parameters. The number of extra parameters is equal to the total number of channels, which is negligible when considering the total number of weights.
➔ we expect no extra risk of overfitting.
- The gradient of a<sub>i</sub> for one layer is :
    <img src='https://user-images.githubusercontent.com/57218700/145714212-70889284-018b-454f-b03e-42af7ec07941.png' width=30%>  <img src='https://user-images.githubusercontent.com/57218700/145714218-7e73a447-be5b-447f-a83e-aca0ea97a4b4.png' width=31%> 
- The time complexity due to `PReLU` is negligible for both forward and backward propagation.
- We adopt the momentum method when updating a<sub>i</sub> (&mu; : momentum, &epsilon; : learning rate) :
    <img src='https://user-images.githubusercontent.com/57218700/145714273-a781eb5c-b620-4fdd-b8c9-c150d7d74ec6.png' width=30%>
- We do not use weight decay when updating a<sub>i</sub>. A weight decay tends to push a<sub>i</sub> to zero, and thus biases `PReLU` toward ReLU.

### Initialization of filter weights for rectifiers
- Bad initialization in rectifier networks can hamper the learning of a highly non-linear system.
- Xavier initialization (Glorot and Bengio)
  - adopt a properly scaled uniform distribution for initialization.
  ➔ Its derivation is based on the assumption that the activations are linear. (invalid for ReLU and `PReLU`)
#### `He initialization`
- **Forward propagation case** : For a conv layer, y<sub>l</sub> = W<sub>l</sub>x<sub>l</sub> + b<sub>l</sub>.
  - Let the initialized elements in W<sub>l</sub> and x<sub>l</sub> be  i.i.d. x<sub>l</sub> and W<sub>l</sub> are independent of each other.
    <img src='https://user-images.githubusercontent.com/57218700/145714557-531d20a0-7706-4933-9c1b-badb232ce751.png' width=30%>  <img src='https://user-images.githubusercontent.com/57218700/145714572-db98d1fb-3ee5-4b17-91ce-b1f09b409b41.png' width=33%>
  - Let W<sub>l-1</sub> have a symmetric distribution around zero and b<sub>l-1</sub> = 0, then y<sub>l-1</sub> has zero mean and has a symmetric distribution around zero.
  - When f is ReLU, E[x<sup>2</sup><sub>l</sub>] = (1/2) * Var[y<sub>l-1</sub>].
    <img src='https://user-images.githubusercontent.com/57218700/145714697-16e619df-b227-4236-9da7-b499b0c34f56.png' width=40%>
  - Key to the initialization design :
    <img src='https://user-images.githubusercontent.com/57218700/145714753-4b0c0b83-94b8-4e1f-bafc-a85314c2dcba.png' width=40%>
  - Sufficient Condition :
    <img src='https://user-images.githubusercontent.com/57218700/145714796-b142b39a-cb68-41c1-b870-33a9fb750989.png' width=30%>

- **Backward propagation case** : &Delta;x<sub>l</sub> = W&#770;<sub>l</sub>&Delta;y<sub>l</sub>, &Delta;y<sub>l</sub> = f'(y<sub>l</sub>)&Delta;x<sub>l+1</sub>.
  - We assume that f′(y<sub>l</sub>) and ∆x<sub>l+1</sub> are independent of each other.
    <img src='https://user-images.githubusercontent.com/57218700/145715252-787bed0f-3b2e-4d81-8f32-de10b117aaa4.png' width=40%>  <img src='https://user-images.githubusercontent.com/57218700/145715268-afb1002c-6647-42bf-9c1d-0085f64f29de.png' width=38%>
  - Key to the initialization design :
    <img src='https://user-images.githubusercontent.com/57218700/145715283-d1bfdbe9-04ed-490c-ae34-fedcf81cbf89.png' width=40%>
  - Sufficient Condition :
    <img src='https://user-images.githubusercontent.com/57218700/145715302-ee70931a-0414-4eea-9cf2-0fc5bac28f19.png' width=30%>
- Convergence of a 22-layer / 30-layer (both ReLU)
    <img src='https://user-images.githubusercontent.com/57218700/145715424-fd5d0cd6-6e45-46f5-b316-8697ce549873.png' width=80%>
  - `He initialization` is able to make the extremely deep model converge.
  -  Xavier method completely stalls the learning, and the gradients are diminishing as monitored in the experiments.
- Our attempts of extremely deep models have not shown benefits on accuracy.

### Experiments
- Comparisions between ReLU and `PReLU` on ImageNet
    <img src='https://user-images.githubusercontent.com/57218700/145715558-fe0dcdbb-46f5-4c82-9558-b2732b31969f.png' width=60%>
  - `PReLU` converges faster than ReLU. Moreover, `PReLU` has lower train error and val error than ReLU.
- Multi-model results for ImageNet test set
    <img src='https://user-images.githubusercontent.com/57218700/145715605-298b9f87-6786-41ea-a807-29233c7e1276.png' width=45%>
  - Our result (4.94%) exceeds the reported human-level performance (5.1%) and GoogleNet.