## On the importance of initialization and momentum in deep learning
- Authors : Sutskever, Ilya and Martens, James and Dahl, George and Hinton, Geoffrey
- Journal : PMLR
- Year : 2013
- Link : http://proceedings.mlr.press/v28/sutskever13.pdf

### Abstract
- SGD with momentum + well-designed random initialization + slowly increasing for momentum parameter  
➔ It can train to levels of performance with only Hessian-Free optimization.
- Carefully tuned momentum methods suffice for dealing with the curvature issues in DNN and RNN without the need for sophisticated second-order methods.

### Introduction
- HF method of Martens (2010) effectively trains RNNs on artificial problems that exhibit very long-range dependencies.
- Contributions : Study effectiveness of SGD when combined with well-chosen initialization schemes and various forms of momentum-based acceleration  
➔ Definite performance gap between plain SGD and HF on certain deep and temporal learning problems  
➔ Show how certain carefully designed schedules for the constant of momentum μ

### Classical Momentum (CM)
- Technique for accelerating gradient descent that accumulates a velocity vector in directions of persistent reduction in the objective across iterations.
- &epsilon; : learning rate, &mu; in [0,1] : momentum coefficient
    <img src="https://user-images.githubusercontent.com/57218700/145682036-e8d3355e-170f-46bb-95b4-e63cc6daca8b.png" width="50%">
- Since directions d of low-curvature have slower local change in their rate of reduction, `CM` can considerably accelerate convergence to a local minimum requiring fewer iterations than steepest descent to reach the same level of accuracy.

### Nesterov’s Accelerated Gradient (NAG)
- `NAG` is a first-order optimization method with better convergence rate guarantee than GD in certain situations.
    <img src="https://user-images.githubusercontent.com/57218700/145682142-7faee9ad-70f1-4f0d-9641-9d129a0cdfdc.png" width="50%">
    <img src="https://user-images.githubusercontent.com/57218700/145682155-2a99f2e8-4b3d-4f05-978f-e7fe438cbcc2.png" width="50%">

### The relationship between CM and NAG
- Both `CM` and `NAG` compute the new velocity by applying a gradient-based correction to the previous velocity vector and then add the velocity to &theta;<sub>t</sub>.
- `CM` computes the gradient update from the current position &theta;<sub>t</sub>.
- `NAG` first performs a partial update to &theta;<sub>t</sub>, computing &theta;<sub>t</sub> + &mu;v<sub>t</sub> (&mu;v<sub>t</sub> is similar to &theta;<sub>t+1</sub>.
- `NAG` changes v in a quicker and more responsive way, letting it behave more stably than `CM` especially for higher values of μ.
#### Where the addition to &mu;v<sub>t</sub>
- Results in an immediate undesirable increase in the objective f. 
- The gradient correction to the velocity v<sub>t</sub> is computed at &theta;<sub>t</sub> + &mu;v<sub>t</sub>.
- If &mu;v<sub>t</sub> is indeed a poor update, then &nabla; f(&theta;<sub>t</sub> + &mu;v<sub>t</sub>) will point back towards more strongly than &nabla; f(&theta;<sub>t</sub>) does.  
➔ provide a larger and more timely correction to v<sub>t</sub> than `CM`.

### Result
- `NAG` can achieve results that are comparable with some of the best HF results for training deep autoencoders.