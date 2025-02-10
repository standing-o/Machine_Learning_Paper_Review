## Attention as an RNN
- Author: Feng et al.
- Journal: arXiv
- Year: 2024
- Link: https://arxiv.org/pdf/2405.13956


### Abstract
- `Transformer` is a powerful architecture for sequence modeling that utilizes GPU parallelism but is **computationally expensive** during inference, making it less useful for low-resource devices like mobile phones.
  - High computational costs limit their use in resource-constrained environments.
  - They cannot efficiently update their output with new tokens, a crucial requirement for many applications.

- **Key Contributions**
  - **Attention as RNN** | The authors demonstrate that the attention mechanism can be interpreted as a specific type of Recurrent Neural Network (RNN) that efficiently handles many-to-one outputs. 
  - **Transformers as RNN Variants** | They show that popular models like Transformers can also be considered as RNNs, but with limitations in updating with new tokens.
  - **New Method for Attention** | Introduces a method to compute attention outputs as many-to-many RNN outputs using the parallel prefix scan algorithm, which improves efficiency.
  - **Aaren Module** | They developed "Aaren," a new attention-based module that maintains the advantages of parallel training (like Transformers) and the efficient token updates of traditional RNNs. 
    - It also requires constant memory storage during inference.
    - Aaren is shown to achieve performance comparable to Transformers across 38 datasets in four major areas: reinforcement learning, event forecasting, time series classification, and forecasting, while being more efficient in both time and memory.


### Introduction
- **Advancements in Sequence Modelling**
  - **Applications** | Sequence modelling is critical for various fields such as
  - **Reinforcement Learning** | Examples include robotics and autonomous driving.
  - **Time Series Classification** | Applications like financial fraud detection and medical diagnoses.
  - **Time Series Forecasting** | Examples include weather predictions and energy consumption forecasts.

- **Transformers' Dominance**
  - Transformer models, introduced by Vaswani et al. in 2017, are noted for their strong performance and ability to utilize GPU parallelism effectively.
  - Over time, many survey papers have focused on Transformer models for sequential tasks across fields like reinforcement learning, time series, and speech processing.

- **Challenges with Transformers**
  - Transformers exhibit quadratic scaling in terms of memory and computation, making them unsuitable for low-resource environments (e.g., battery-powered devices).
  - Requires linear memory based on the number of tokens.
  - Needs caching of all previous tokens, particularly problematic with long contexts found in time series data.

- **Understanding Attention in Transformers**
  - The paper presents attention as a type of Recurrent Neural Network (RNN) that can compute its outputs efficiently. 
  - Limitations of Popular Models: Models like Transformers and Perceivers are viewed as RNNs but struggle to update efficiently with new tokens, which is crucial for sequential data processing.

- **Aaren**
  - A newly proposed method uses the parallel prefix scan algorithm to compute attention as a many-to-many RNN output.
  - Aaren can:
    - Be trained in parallel (similar to Transformers).
    - Update with new tokens efficiently, using only constant memory for inferences (as traditional RNNs do).
  - Empirical results show that Aaren achieves performance levels comparable to Transformers across 38 datasets in popular sequential data tasks such as reinforcement learning, event forecasting, time series classification, and forecasting.


### Background
#### Recurrent Neural Networks (RNNs)
- RNNs are designed for sequence modeling, processing data sequentially.
- RNNs compute hidden states iteratively:
$h_t = f_\theta(h_{t-1}, x_t)$

- where:
  - $\( h_t \)$ | The hidden state at time $\( t \)$.
  - $\( h_{t-1} \)$ | The hidden state from the previous time step.
  - $\( x_t \)$ | The input (token) at time $\( t \)$.
  - $\( f_\theta \)$ | A function (neural network) parameterized by $\( \theta \)$.


- Initial State | $\( h_0 \)$ is usually learned through backpropagation.
- Variants | LSTMs and GRUs are common types of RNNs:
  - They maintain a constant hidden state size, leading to efficient memory use during inference.
- Limitations | Traditional RNNs face scalability issues due to their inherently iterative nature and lack of parallel processing.

#### Attention
- Function | Attention allows the retrieval of information from various context tokens for specific query tokens:
$\text{Attention}(Q, K, V) = \text{softmax}(QK^T)V$
- where:
  - $\( Q = X_QW_q \)$ | Query matrix derived from query tokens.
  - $\( K = X_CW_k \)$ | Key matrix derived from context tokens.
  - $\( V = X_CW_v \)$ | Value matrix derived from context tokens.
  - $\( W_q, W_k, W_v \in \mathbb{R}^{d \times d} \)$ | Learnable weight matrices.

- The softmax function normalizes the dot products to compute weights for a weighted average.


- **Key Features**
  - Non-iterative design aids parallelization on GPUs.
  - Transformers utilize self-attention, where queries match the context.
  - Performance Cost: Attention has a quadratic computation cost concerning the number of tokens, limiting efficient updating as new tokens are added.

### Methodology
- Attention as RNN | The paper provides insights into viewing attention mechanisms as special types of Recurrent Neural Networks (RNNs). 
    <img src='https://github.com/user-attachments/assets/6f6d3cd4-030a-473e-abbc-d71856130225' width=80%>
  - (a) illustrates the conventional method of computing attention, emphasizing that it computes the final output in a manner akin to a "many-to-one" RNN. In this case, multiple context tokens are processed to yield a single output.
  - (b) shows the Transformer's self-attention method, where input tokens are treated as the initial hidden states of the RNN. 
    - This illustrates how it maintains a similar relationship to RNNs.
  - (c) displays the Perceiver’s cross-attention, which uses context-dependent latent variables as initial hidden states, again relating it back to RNN characteristics.

- **Efficiency of Attention**
  - The paper proposes a new efficient method that allows attention to be handled as a "many-to-many" RNN. 
  - This comes from utilizing the parallel prefix scan algorithm which enhances the efficiency in processing new tokens sequentially while maintaining low computation costs.

- **Aaren Module**
  - The introduction of Aaren (Attention as a Recurrent Neural Network) aims to provide a module that can:
    - Be trained in parallel (like Transformers).
    - Be updated efficiently at inference time with new tokens, using constant memory (similar to traditional RNNs).
  - This methodology is expected to perform comparably to Transformers across various datasets and tasks, while being more efficient in terms of time and memory requirements.


#### Attention as a (many-to-one) RNN
<img src='https://github.com/user-attachments/assets/83c0d46c-ac41-47c5-8a76-4387ed250396' width=50%>

- **Attention Mechanism**
  - Attention allows a model to focus on relevant parts of the input when generating an output.
  - For a query vector $\( q \)$, it processes $\( N \)$ context tokens $\( x_1^N \)$ using their associated keys $\( k_i \)$ and values $\( v_i \)$.
  - Output Formula:
    - The output \( o_N \) for attention can be expressed as:
$o_N = Attention(q, k_{1:N}, v_{1:N}) = \frac{\sum_{i=1}^{N} \exp(s_i) v_i}{\sum_{i=1}^{N} \exp(s_i)}$
     - where $\( s_i = \text{dot}(q, k_i) \)$.

  - **Components of the Output**
    - **Numerator** | $\( \hat{a}_N = \sum_{i=1}^{N} \exp(s_i) v_i \)$
      - This is a weighted sum of the values \( v_i \), weighted by the exponentiated scores.
    - **Denominator** | $\( \hat{c}_N = \sum_{i=1}^{N} \exp(s_i) \)$
      - This normalizes the weights to ensure they sum to one.

- **Recursive Computation**
  - Attention can be computed iteratively as rolling sums:
    - For $\( k = 1, \ldots, N \)$,
$\hat{a}_k = \hat{a}_{k-1} + \exp(s_k) v_k$
$\hat{c}_k = \hat{c}_{k-1} + \exp(s_k)$
  - This method can become unstable due to numerical precision issues.

- **Stable Implementation**
  - To stabilize the computation, a cumulative maximum $\( m_k = \max_{i \in \{1, \ldots, k\}} s_i \)$ is introduced:
$a_k = \sum_{i=1}^{k} \exp(s_i - m_k) v_i$

$c_k = \sum_{i=1}^{k} \exp(s_i - m_k)$
  - Updates are made as:
$a_k = a_{k-1} \exp(m_{k-1} - m_k) + v_k \exp(s_k - m_k)$
$c_k = c_{k-1} \exp(m_{k-1} - m_k) + \exp(s_k - m_k)$
$m_k = \max(m_{k-1}, s_k)$

- **Structure of Attention as RNN**
  - Each RNN cell computes:
    - **Inputs** | $\( (a_{k-1}, c_{k-1}, m_{k-1}, q) \)$
    - **Outputs** | $\( (a_k, c_k, m_k, q) \)$
  - Initial state is set to $\( (a_0, c_0, m_0, q) = (0, 0, 0, q) \)$.

- **Computational Methods**
  - Attention can be executed:
    - **Sequentially** | token-by-token with $\( O(1) \)$ memory.
    - **In Parallel** | consumes $\( O(N) \)$ memory for complete context.
    - **Block-wise** | processes tokens in chunks using $\( O(b) \)$ memory.


- **Viewing Attention Models as RNNs**
  - Many existing models, like Transformers, can be interpreted as RNNs with attention layers acting as their recurrent states.
  - This insight allows for efficient computation using RNN properties.


#### Attention as a (many-to-many) RNN
- Develop an attention-based model that efficiently performs updates like RNNs.

- **Method**
  - Introduce a parallelized method to compute attention as a many-to-many RNN. Key elements of this method include:
  - Using the parallel prefix scan algorithm (Blelloch, 1990) to compute N prefix computations using an associative operator, $⊕$.
  - Formulating the computation of attention outputs as:
$o_i = \text{Attention}(q, x_{1:i}) \quad \text{for } i = 1, \ldots, N$

- **Key Components**
  - Attention output calculation: The output of the attention function is expressed as:
$\text{Attention}(q, x_{1:k}) = o_k = a_k \cdot c_k$
    - where:
      - $\( a_k = \sum_{i=1}^{k} \exp(s_i - m_k)v_i \)$
      - $\( a_k \)$ | Acknowledges the importance of context tokens.
      - $\( s_i \)$ | Represents the score of the dot product between query $\( q \)$ and key $\( k_i \)$.
      - $\( v_i \)$ | Value associated with each key.
      - $\( m_k = \max_{i \in \{1, \ldots, k\}} s_i \)$ is used to help avoid numerical instability.

  - $\( c_k = \sum_{i=1}^{k} \exp(s_i - m_k) \)$
    - $\( c_k \)$ | Normalizes the attention value by scaling.

- **Parallel Computation**
  - The proposed associative operator $\( \oplus \)$ evaluates and combines three variables:
    - $\( m_A \)$ | The maximum score from set A.
    - $\( u_A = \sum_{i \in A} \exp(s_i - m_A) \)$ | Total importance found within the context.
    - $\( w_A = \sum_{i \in A} \exp(s_i - m_A)v_i \)$ | Aggregated values from the context.

- **Output of Algorithm**
  - After applying the associative operator, the results yield
    $\{(m_{1,\ldots,k}, u_{1,\ldots,k}, w_{1,\ldots,k})\}_{k=1}^N = \{(m_k, u_k, w_k)\}_{k=1}^N$
  - Finally, we obtain the many-to-many attention outputs with:
$\text{Attention}(q, x_{1:k}) = o_k = a_k \cdot c_k$

<img src='https://github.com/user-attachments/assets/42523119-b365-4aa9-863a-f31fec752018' width=60%>

#### Aaren
- Attention as a Recurrent Neural Network
- **Aaren** stands for Attention as a Recurrent Neural Network. It builds on the traditional attention mechanism using a many-to-many RNN formulation. 
- Input and Output
  - **Input** | Takes N inputs (context tokens).
  - **Output** | Produces N outputs, where the i-th output is derived from the first i inputs. Each output aggregates information from the previous inputs up to that point.

- Aaren is naturally **stackable**, meaning you can build layers of Aaren modules on top of each other.
- It computes individual loss terms for each token in the sequence.

- **Query Mechanism**: 
  - In Aaren, the query token $(q)$ is not simply one of the input tokens. Instead, it is learned during training through back-propagation, enhancing the model’s adaptability.

- Unlike Transformers, which utilize causal self-attention, Aaren leverages many-to-many RNN computations that are more efficient.
- Aaren requires constant memory for processing, as opposed to Transformers, which demand linear memory during inference. It does not need to store all previous tokens, allowing it to process data more swiftly and economically.

- **Functioning**
  - **Initialization** | The initial state of the system is represented as $h(0)$ for the inputs: $\( h(0)_1, h(0)_2, \ldots, h(0)_N \)$ corresponds to input tokens $\( x_1, x_2, \ldots, x_N \)$.
  - **Recursion** | The next hidden states are computed by evaluating each output using previous hidden states and the current input token: $[h(j+1)_1, \ldots, h(j+1)_N] \gets Aaren(q(j), [h(j)_1, \ldots, h(j)_N])$
    - This shows that Aaren recycles previous outputs along with the current query to update hidden states.

<img src='https://github.com/user-attachments/assets/b4ebbfb9-c070-460b-b307-433b2d49c044' width=60%>


### Experiments
- Compare performance and resource usage (time and memory) of Aarens with Transformers.
- **Problem Settings**
  - Experiments are conducted across four areas:
    - Reinforcement Learning, Event Forecasting, Time Series Forecasting and Time Series Classification

- A total of 38 datasets were evaluated
  - Each dataset was evaluated using 5 different random seed initializations.

- **Model Comparison**
  - Aarens was compared to Transformers directly by replacing Transformers in specialized models:
    - Reinforcement Learning used Decision Transformer (Chen et al., 2021).
    - Event Forecasting compared with Transformer Hawkes Process (Zuo et al., 2020; Bae et al., 2023).
    - Time Series Forecasting leveraged a Transformer with normalized input (Liu et al., 2022).
    - Time Series Classification employed a basic causal transformer from Wu et al. (2023).

- **Implementation Details**
  - All experiments were based on established libraries and codes:
  - Reinforcement learning: Code by Barhate (2022).
  - Time series tasks: Time Series Library (Wu et al., 2023).

- Aarens and Transformers share the same interfaces and hyperparameters to ensure fairness in comparison.


#### Reinforcement Learning
- **Context of Experiments**
  - The experiments aim to compare the performance of two models: Aarens and Transformers in the field of reinforcement learning (RL).
  - RL is a method where models learn to make decisions by interacting with an environment and receiving feedback in the form of rewards.
  - The models' goal is to train a policy, which is basically a strategy to select the right actions based on previous experiences and rewards.

- Each dataset is made up of 1 million timesteps, which are interactions between the model and the environment.
- In total, the performance across 12 datasets (4 environments x 3 datasets each) was compared.
- Aarens were found to achieve competitive results against Transformers across all twelve datasets.


- **Computational Efficiency**
  - Aarens can process new interactions with constant computation, making them more suitable for reinforcement learning compared to Transformers, which can be computationally expensive.

#### Event Forecasting
- Models the probability distribution of the next event time and its label from a sequence of irregularly spaced discrete events over time.
- Applications include finance (transactions), healthcare (patients' observations), and e-commerce (purchases).

- Aaren (a new model type) is compared with Transformers (a popular model type) in the context of EF.
- Transformers are replaced in the Transformer Hawkes Process (an established model for EF) for this comparison.

- Eight popular benchmarking datasets were used for the experiments, with seven being real-world datasets (MIMIC, Wiki, Reddit, Mooc, StackOverflow, Uber, Taxi) and one synthetic dataset (Sin).

- Aaren performed comparably to Transformers across all the datasets evaluated.
- Significantly, Aaren can efficiently process new inputs as they arrive—which is beneficial in EF scenarios where events occur irregularly.

- Fewer metrics show that Aaren often outperforms or closely matches Transformers while being more computationally efficient.

<img src='https://github.com/user-attachments/assets/93debeb3-fbab-47c2-9da4-a1cea31124c7' width=60%>
<img src='https://github.com/user-attachments/assets/b440962f-d92f-4003-a46e-cc9f19feaf64' width=60%>


#### Time Series Forecasting
- Compare the performance of Aarens and Transformers in time series forecasting.
- Predict T future values of a continuous series based on past observations.
- 8 real-world datasets (Weather, Exchange, Traffic, ECL, ETTh1, ETTh2, ETTm1, and ETTm2).

- **Performance Metric**
  - MSE (Mean Squared Error) | Lower values are better; indicates how close the predicted values are to the actual values.
  - MAE (Mean Absolute Error) | Lower values are better; indicates the average error magnitude in predictions.

- **Performance** | Aarens perform comparably with Transformers across all datasets for time series forecasting.
  - Efficiency | Aarens demonstrate an efficient processing capability, which is beneficial for time series-related tasks.
Evaluation Setup: Both models were evaluated with T values ranging from 96, 192, 336, to 720, with the focus shifted to results for T = 192 due to space limitations.
    <img src='https://github.com/user-attachments/assets/34569b6f-8a69-4b70-95c1-052ce793e4de' width=60%>


#### Time Series Classification
- The experiments aimed to evaluate how well Aarens perform against Transformers in TSC tasks where the goal is to predict the label of time series data.
- TSC is important for various real-world applications, such as:
  - Pattern Recognition: Identifying patterns in data like electrocardiograms.
  - Anomaly Detection: For example, detecting bank fraud.
  - Failure Prediction: Such as predicting fluctuations in power grids.

- The results indicated that Aarens performed comparably to Transformers across all the tested datasets.   
- This implies that Aarens can achieve similar levels of accuracy and reliability in identifying labels from time series data.


#### Analyses
- **Memory Complexity**
  - Transformers (with KV-caching): Memory usage increases linearly with the number of tokens during inference.
  - Aaren: Uses constant memory regardless of the number of tokens, making it far more memory-efficient.


- **Time Complexity**
  - Transformers | Cumulative time needed to process tokens is quadratic (O(N^2)), where N is the number of tokens. This is derived from the formula:
$O(1 + 2 + ... + N) = O\left(\frac{N(N + 1)}{2}\right) = O(N^2)$
    - which equates to quadratic complexity because of the summation of sequential integers.
  - Aaren  processes tokens with cumulative time complexity that is linear (O(N)), demonstrating a significant efficiency advantage.

- Number of Parameters
  - Aaren requires slightly more parameters than Transformers due to learning an initial hidden state $\( q \)$.
    - Transformers Parameters: 3,152,384
    - Aaren Parameters: 3,152,896
  - The increase is marginal (~0.016%), which is deemed a minor trade-off against the memory and time efficiency gains.

<img src='https://github.com/user-attachments/assets/8a615f0c-a465-4b4d-bcfa-aa896078610d' width=60%>


### Conclusion
- The paper demonstrates that attention can be viewed as a special type of Recurrent Neural Network (RNN) designed to efficiently compute outputs from sequences of inputs.
- Traditional attention mechanisms (like those in Transformers) cannot easily update their outputs when new tokens (data) are introduced. This limits their utility in dynamic environments.
- The authors propose a method for computing attention in a many-to-many fashion using the parallel prefix scan algorithm. This allows updates with new tokens efficiently.

- **Aaren Module**
  - Parallel Training: Aaren can be trained in parallel (similar to Transformers).
  - Constant Memory Use: At inference time, it only requires a constant amount of memory (similar to traditional RNNs).

- Aaren was tested on 38 datasets across four areas: reinforcement learning, event forecasting, time series classification, and time series forecasting. 
  - The results showed that Aaren competes well with Transformers while being more efficient in time and memory consumption.
