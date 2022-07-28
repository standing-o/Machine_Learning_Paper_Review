## Training very deep networks
- Authors : Srivastava, Rupesh Kumar and Greff, Klaus and Schmidhuber, Jürgen
- Journal : arXiv preprint arXiv
- Year : 2015
- Link : https://arxiv.org/pdf/1507.06228.pdf

### Abstract
- Training becomes more difficult as depth increases.
- `Highway networks` allow unimpeded information flow across many layers on information highways.
  - use adaptive **gating units** to regulate the information flow
  - Even with hundreds of layers, highway networks can be trained directly through simple gradient descent.

### Introduction
- LSTM-inspired adaptive gating mechanism that allows for computation paths along which information can flow across many layers without attenuation.
➔ information highways
- Extremely deep `highway networks` can be trained directly using SGD, in contrast to plain networks which become hard to optimize as depth increases.

### `Highway Networks`
1. A plain feed forward neural network : y = H(x, W<sub>H</sub>) (H : nonlinear transformation)
2. `Highway Networks`
  - T : transform gate, C : Carry gate, We set C = 1-T,
    <img src='https://user-images.githubusercontent.com/57218700/145711842-9a06e28a-0cb3-44c1-8ade-26169ad769c5.png' width=50%>
  - `Highway network` consists of multiple blocks such that the i-th block computes a block state H<sub>i</sub>(𝑥) and transform gate output T<sub>i</sub>(𝑥).
    <img src='https://user-images.githubusercontent.com/57218700/145711906-4b70339f-301d-49a1-873e-2e17768377a8.png' width=50%>
  - The block out output y<sub>i</sub> = H<sub>i</sub>(x) * T<sub>i</sub>(x) + x<sub>i</sub> * (1-T<sub>i</sub>(x)), which is connected to the next layer.

#### Training Deep `Highway Networks`
- Transform gate defined as T(x) = 𝜎(W<sub>T</sub>x + b<sub>T</sub>), where W<sub>T</sub> is the weight matrix and b<sub>T</sub> the bias vector for the transform gates.
➔ Negative bias initialization for the transform gates was sufficient for training to proceed in very deep networks for various zero-mean initial distributions of W<sub>H</sub> and different activation functions used by H.

### Experiments
- Plain networks become much harder to optimize with increasing depth, while `highway networks` with up to 100 layers can still be optimized well.
    <img src='https://user-images.githubusercontent.com/57218700/145712911-569e7ee9-d7c0-4601-832b-bce6aeb8c281.png' width=90%>
- Test set classification accuracy on MNIST
    <img src='https://user-images.githubusercontent.com/57218700/145712934-c7a85410-6750-40eb-98ba-a7bf6f405141.png' width=80%>

### Analysis of results
- Visualization of best 50 hidden-layer `highway networks` trained on MNIST and CIFAR-100
    <img src='https://user-images.githubusercontent.com/57218700/145712481-aaa0c784-45be-46ca-88a0-ec2fb7fa722c.png' width=80%>
  - Column 1 : Most biases decreased further during training. For the CIFAR-100 network the biases increase with depth forming a gradient.
  - Column 2 : The strong negative biases at low depths are not used to shut down the gates, but to make them more selective.
  - Column 3 : The transform gate activity for a single example is very sparse.
  - Column 4 : Most of the outputs stay constant over many layers forming a pattern of stripes and most of the change in outputs happens in the early layers.
  - Row 2 : Routing of Information
  ➔ The network can learn to dynamically adjust the routing of the information based on the current input.

- Visualization showing the extent to which the mean transform gate activity for certain classes differs from the mean activity over all training samples.
  ➔ For digits 0 and 7 substantial differences can be seen within the first 15 layers.
  <img src='https://user-images.githubusercontent.com/57218700/145712757-e0136f58-0d90-4be3-b11f-81a16ff56971.png' width=50%>
- Layer Importance
  ➔ For complex problems a `highway network` can learn to utilize all of its layers, while for simpler problems like MNIST it will keep many of the unneeded layers idle.
  <img src='https://user-images.githubusercontent.com/57218700/145712834-c32eff46-baaf-491c-818b-5e3909eedeef.png' width=80%>