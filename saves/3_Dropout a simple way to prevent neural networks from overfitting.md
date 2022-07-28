## Dropout: a simple way to prevent neural networks from overfitting
- Authors : Srivastava, Nitish and Hinton, Geoffrey and Krizhevsky, Alex and Sutskever, Ilya and Salakhutdinov, Ruslan
- Journal : JMLR
- Year : 2014
- Link : https://www.cs.cmu.edu/~rsalakhu/papers/srivastava14a.pdf

### Abstract
- The key idea of `Dropout` is to randomly drop units from the neural network during training.
➔ This prevents units from co-adapting too much.
➔ During training, dropout samples from an exponential number of different thinned networks.
➔ At test time, it is easy to approximate the effect of averaging the predictions of all these thinned networks by simply using a single unthinned network that has smaller weights.

### Introduction
- Approximating an equally weighted geometric mean of the predictions of an exponential number of learned models that share parameters.
- Addressing overfitting and model combination issues :
➔ `Dropout` prevents overfitting and provides a way of approximately combining exponentially many different NN architectures efficiently. 
- `Dropout` = dropping out units in NN
- The choice of which units to drop is random 🠊 Each unit is retained with a fixed probability **p** independent of other units.
- The thinned network consists of all the units that survived dropout NN with n units -> collection of 2^𝑛 possible thinned NNs 
 (parameters : O(𝑛<sup>2</sup>))
    <img src='https://user-images.githubusercontent.com/57218700/145707370-1e6dbc42-f288-45af-83ce-67ce3ba3b3bd.png' width=50%>
- At **test time**, it is not feasible to explicitly average the predictions from exponentially many thinned models.
  ➔ approximate averaging method using a single neural net at test time without dropout 
  ➔ If a unit is retained with probability p during training, the outgoing weights of that unit are multiplied by p at test time.
    <img src='https://user-images.githubusercontent.com/57218700/145707444-ccb89c0d-8d64-423d-b938-b3ccab723ab5.png' width=50%>

### Model Description
- Feed forward with dropout
    <img src='https://user-images.githubusercontent.com/57218700/145707480-915eea3f-17f9-463d-9ce1-d575f3eb4ff6.png' width=40%>

### Learning `Dropout` Nets
#### Back-propagation
- Forward and backpropagation for that training case are done only on this thinned network.
- The gradients for each parameter are averaged over the training cases in each mini-batch.

#### Max-norm regularization
- Useful for `dropout` : constraining the norm of the incoming weight vector at each hidden unit to be upper bounded by a fixed constant c (hyperparameter) 
- The NN was optimized under the constraint ||𝑾||<sub>2</sub>≤ c.
➔ This constraint was imposed during optimization by projecting W onto the surface of a ball of radius c, whenever W went out of it.

#### Unsupervised Training
- NN can be pretrained using stacks of RBM or autoencoders, DBM The weights obtained from pretraining should be scaled up by a factor of 1/p.
➔ This makes sure that for each unit, the expected output from it under random `dropout` will be the same as the output during pretraining.
- The stochastic nature of `dropout` might wipe out the information in the pretrained weights 🠊 choose learning rates smaller.

### Experimental Results
- Test error on MNIST (The networks have 2 to 4 hidden layers each with 1024 to 2048 units.)
    <img src='https://user-images.githubusercontent.com/57218700/145707840-ae1cd834-6824-4fe7-a06f-b47a71a6543e.png' width=50%>
#### Sailent features learned by an autoencoder on MNIST
- Effect on features
➔ The features without `dropout` have co-adapted in order to produce good reconstruction.
➔ Each hidden unit on an autoencoder with `dropout` seems to be detecting a meaningful feature.
➔ `Dropout` does break up co-adaptations, which is probably the main reason why it leads to lower generalization errors.
- Effect on Sparsity
➔ The activation of the hidden units become sparse
- Effect on `dropout` rate
➔ optimal value of p : 0.4 ≤ p ≤ 0.8 (0.5, 0.6)
- Effect of Data Set Size
➔ The number of data needs to be 1K.
➔ As the size of dataset increases, the effect does not drastic.