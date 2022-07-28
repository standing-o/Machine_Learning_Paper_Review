## Attention is all you need
- Authors : Vaswani, Ashish, et al.
- Journal : Advances in neural information processing systems
- Year : 2017
- Link : https://arxiv.org/abs/1706.03762

###  Abstract
- `Transformer` based solely on `attention` mechanisms, dispensing with recurrence and convolutions entirely.

### Introduction
- **Recurrent models** typically factor computation along the symbol positions of the input and output sequences. 
➔ Aligning the positions to steps in computation time, they generate a sequence of hidden states h<sub>t</sub>, as a function of the previous hidden state h<sub>t−1</sub> and the input for position t. 
➔ This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.
- `Transformer`, a model architecture eschewing recurrence and instead relying entirely on an `attention` mechanism to draw global dependencies between input and output.
  ➔ more parallelization
  <img src='https://user-images.githubusercontent.com/57218700/167640992-326140ff-4362-4bda-9ad8-26b2da4be422.png' width=50%>


### Encoder and Decoder stacks
#### **Encoder:**
 - Stack of N = 6 identical layers. Each layer has two sub-layers. 
➔ The first is a multi-head self-`attention` mechanism, and the second is a simple, position-wise fully connected feed-forward network. ➔ A residual connection around each of the two sub-layers, followed by layer normalization. (LayerNorm(x + Sublayer(x)))

#### **Decoder:**
- Stack of N = 6 identical layers. 
➔ In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head `attention` over the output of the encoder stack.
➔ Residual connections around each of the sub-layers, followed by layer normalization.
➔ **Masking** ensures that the predictions for position i can depend only on the known outputs at positions less than i.

### `Attention`
- Mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.
-  The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

####  Scaled Dot-Product `Attention`
- The input consists of queries and keys of dimension d<sub>k</sub>, and values of dimension d<sub>v</sub>.
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^{T}}{\sqrt{d_{k}}})V" width=40%>  
  <img src='https://user-images.githubusercontent.com/57218700/167627820-279475ac-c220-46fc-aa8e-edff15eef9a2.png' width=20%>

#### Multi-head `Attention`
- Instead of performing a single `attention` function with dmodel-dimensional keys, values and queries, 
➔ It beneficial to linearly project the queries, keys and values h times with different, learned linear projections to d<sub>k</sub>, d<sub>k</sub> and d<sub>v</sub> dimensions, respectively.
- Multi-head `attention` allows the model to jointly attend to information from different representation subspaces at different positions.
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;\text{MultiHead}(Q,K,V)=\text{Concat}(head_1,...,head_h)W^O" width=60%>
  where <img src="https://latex.codecogs.com/svg.latex?\Large&space;\text{head}_i=\text{Attention}(QW^{Q}_{i},KW^{K}_{i},VW^{V}_{i})" width=40%>
  the projections are parameter matrices:
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;W^Q_i\in\mathbb{R}^{d_\text{model}\times{d_k}},W^K_i\in\mathbb{R}^{d_\text{model}\times{d_k}},W^V_i\in\mathbb{R}^{d_\text{model}\times{d_v}}" width=60%> and <img src="https://latex.codecogs.com/svg.latex?\Large&space;W^O\in\mathbb{R}^{{d_k}\times{d_\text{model}}}" width=20%>
   <img src='https://user-images.githubusercontent.com/57218700/167632067-a51f8405-fa6f-46cb-82cc-da80d6b8ba3e.png' width=30%>

### Applications of `Attention` in our model
#### 1. Encoder-decoder `attention` layers
- The queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder.
  ➔ Every position in the decoder to attend over all positions in the input sequence
   <img src='https://user-images.githubusercontent.com/57218700/167633336-380e83c6-222d-455a-9872-5889c31906ac.png' width=40%>

#### 2. The **encoder** contains self-`attention` layers.
- In a self-`attention` layer all of the keys, values and queries come from the same place
  ➔ The output of the previous layer in the encoder. 
  ➔ Each position in the encoder can attend to all positions in the previous layer of the encoder.
   <img src='https://user-images.githubusercontent.com/57218700/167632819-a9b3ccd2-064a-4816-835b-9383d23f9d51.png' width=20%>

#### 3. The **decoder** contains self-`attention` layers.
- Each position in the decoder to attend to all positions in the decoder up to and including that position.
- (To preserve the auto-regressive property) Scaled dot-product `attention` by masking out (setting to −∞) all values in the input
  of the softmax which correspond to illegal connections.
   <img src='https://user-images.githubusercontent.com/57218700/167632934-1c3a1c62-60ee-4f6b-ae50-8da528f06133.png' width=20%>

### Position-wise Feed-Forward Networks
- Each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically.
  ➔ Two linear transformations with a ReLU activation in between:
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;\text{FFN}(x)=max(0,xW_{1}+b_{1})W_{2}+b_{2}" width=40%>
- While the linear transformations are the same across different positions, they use different parameters from layer to layer.

### Embeddings and Softmax
- Learned embeddings to convert the input tokens and output tokens to vectors of dimension d<sub>model</sub>.
-  The usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.
-  In the embedding layers, we multiply those weights by d<sub>model</sub><sup>1/2</sup>.

### Positional Encoding
- Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence,
  ➔ **Positional encodings** to the input embeddings at the bottoms of the encoder and decoder stacks. 
  ➔ Same dimension d<sub>model</sub> as the embeddings ➔ the two can be summed.
  
- We use sine and cosine functions of different frequencies:
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;\text{PE}_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})" width=40%>
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;\text{PE}_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})" width=40%>
  where pos is the position and i is the dimension. The wavelengths form a geometric progression from 2π to 10000 · 2π.
  ➔ We hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, PE<sub>pos+k</sub> can be represented as a linear function of PE<sub>pos</sub>.

### Training
- **Dataset** : the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs
- **Optimizer** :  Adam optimizer & We varied the learning rate over the course of training:
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;lrate=d^{-0.5}_{model}\cdot{min(\verb|step_num|^{-0.5},\verb|step_num|\cdot{\verb|warmup_steps|^{-1.5}}})" width=80%>
  ➔ This corresponds to increasing the learning rate linearly for the first "warmup_steps" training steps, and decreasing it thereafter proportionally to the inverse square root of the step number.
- **Regularization**
➔ Residual Dropout : dropout<sup>1</sup> to the output of each sub-layer, before it is added to the sub-layer input and normalized and dropout<sup>2</sup> to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.
➔ Label Smoothing
- **Results** : The `Transformer` achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.
  <img src="https://user-images.githubusercontent.com/57218700/167639125-4828c9d3-cb9d-425c-9164-df92d8955e26.png" width=70%>

### Conclusion
- `Transformer`, the first sequence transduction model based entirely on `attention`, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-`attention`.