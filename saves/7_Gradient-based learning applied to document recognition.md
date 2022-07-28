## Gradient-based learning applied to document recognition
- Authors : LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick
- Journal : IEEE
- Year : 1998
- Link : https://ieeexplore.ieee.org/document/726791

### Introduction
- Machine learning with NN : hand-designed heuristics + automatic learning
- Pattern recognition

**1. First module : feature extractor**
  - Input patterns 🠊 low dimensional vectors
  - Distortions of the input 🠊 do not change their nature
  - Hand-craft feature extractor 🠊 prior knowledge

**2. Second module : Classifier**
  - General-purpose and trainable
➔ Accuracy is largely determined by the ability of the feature extractor.

### Learning from data
- Z<sup>p</sup> : p-th input pattern, W : parameters, Y<sup>p</sup> : output, D<sup>p</sup> : desired output
  - Learning machine computation : Y<sup>p</sup> = F(Z<sup>p</sup>, W)
  - Loss function : E<sup>p</sup> = D(D<sup>p</sup>, F(W, Z<sup>p</sup>))
  - W is updated : 
    <img src='https://user-images.githubusercontent.com/57218700/145716178-c2011e3b-dcfa-404e-ad8b-86d4eae5b822.png' width=30%>

### Gradient back-propagation
- Usefulness of gradient-based learning was not widely realized until the following three events occurred.
1. The presence of local minima in the loss function does not seem to be a major problem in practice.
2. Efficient procedure = Back-propagation to compute the gradient in a non-linear system composed of several layers of processing.
3. Back-prop applied to multi-layer neural networks can solve complicated learning tasks.

### Globally trainable systems
- Most practical pattern recognition systems are composed of multiple modules.
➔ Each module must be continuous and differentiable almost everywhere with respect to the internal parameters of the module.
- X<sub>n</sub> = F<sub>n</sub>(W<sub>n</sub>, X<sub>n-1</sub>).
    <img src='https://user-images.githubusercontent.com/57218700/145716368-c60a46ab-af44-4499-9edf-8ae09e125fe7.png' width=30%>

### `Convolutional Networks`
- Ordinary fully connected feedforward network with some success for image recognition, there are problems.
1. Typical images are large, often with several hundred variables (pixels).
➔ Large number of parameters increases the capacity of the system and therefore requires a larger training set.
2. Handwriting ~ size, slant, position variations for individual characters
3. Variables are spatially or temporally nearby are highly correlated.
🠊 Deficiency of fully connected architectures is that the topology of the input is entirely ignored.
#### `LeNet-5`, a `CNN`
- Each plane is a feature map, i.e., a set of units whose weights are constrained to be identical.
    <img src='https://user-images.githubusercontent.com/57218700/145716660-7a03907a-7ce4-4bcc-86f3-a254f0f9fc92.png' width=80%>
- **Local receptive fields**
  - Neurons can extract elementary visual features such as oriented edges, end-points, corners.
  - These features are combined by the subsequent layers in order to detect higher-order features.
- **Shared weights**
  - Units in a layer shares the same set of weights. The set of outputs of the units in such a plane is called a feature map. 
  - Units in a feature map are all constrained to perform the same operation on different parts of the image. 
  - A complete convolutional layer is composed of several feature maps, so that multiple features can be extracted at each location.
- **Sub-sampling**
  -  A simple way to reduce the precision with which the position of distinctive features are encoded in a feature map is to reduce the spatial resolution of the feature map.
  ➔ Reducing the resolution of the feature map and the sensitivity of the output to shifts and distortions.