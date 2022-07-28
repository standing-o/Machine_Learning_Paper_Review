## Time series forecasting with deep learning: A survey
- Authors : Lim, Bryan and Zohren, Stefan
- Journal : Philosophical Transactions of the Royal Society A
- Year : 2021
- Link : https://arxiv.org/pdf/2004.13408.pdf

## Abstract
- Common encoder and decoder designs used in both one-step-ahead and multi-horizon time series forecasting
➔ Describing how temporal information is incorporated into predictions by each model.
- Recent developments in hybrid deep learning models combine well-studied statistical models with neural network components.

## Introduction
- Applications of time-series modelling
➔ climate modelling, biological sciences, medicine, commercial decision making in retail, finance
- Traditional methods
➔ autoregressive(AR), exponential smoothing, structural time series models
- Deep learning
➔ image classification, natural language processing, reinforcement learning
- Modern machine learning methods provide a means to learn temporal dynamics in a purely data-driven manner.
- By incorporating bespoke architectural assumptions, or inductive biases that reflect the nuances of underlying datasets, deep neural networks are able to learn complex data representations
- Deep neural networks are able to learn complex data representations, which alleviates the need for manual feature engineering and model design.
- Automated approaches to time series forecasting
➔ automatic parametric model selection, traditional machine learning methods (kernel regression, SVR)
- Gaussian processes 🠊 deep Gaussian processes, neural processes

## Deep learning architectures for time series forecasting
- Time series forecasting models predict future values of a target y<sub>i,t</sub> for a given entity i at time t. Each entity represents a logical grouping of temporal information.
- One-step-ahead forecasting models:
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{y}_{i,t+1}=f(y_{i,t-k:t},x_{i,t-k:t},s_i)" width=35%>
  ➔ y&#770;<sub>i,t+1</sub> is model forecast
  ➔ y<sub>i,t-k:t</sub>, x<sub>i,t-k:t</sub> are observations of the target and exogenous inputs respectively over a look-back window k, s<sub>i</sub> is static metadata associated with the entity
  ➔ f(.) is the prediction function learnt by the model
  ➔ The same components can be extended to multivariate models WLOG.

### (a) Basic Building Blocks
- Deep neural networks learn predictive relationships by using a series of non-linear layers to construct intermediate feature representations.
➔ Encoding relevant historical information into a latent variable z<sub>t</sub>, with the final forecast produced using z<sub>t</sub> alone (y<sub>i,t</sub> replaced by y<sub>t</sub>):
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;f(y_{t-k:t},x_{t-k:t},s)=g_{\text{dec}}(z_t),\,\,z_t=g_{\text{enc}}(y_{t-k:t},x_{t-k:t},s)" width=60%>
➔ g<sub>enc</sub>(.), g<sub>dec</sub>(.) are encoder and decoder functions respectively.
- Incorporating temporal information using different encoder architectures.
  <img src='https://user-images.githubusercontent.com/57218700/151643632-9092c7e4-8787-48d8-9993-3e9b5166274d.png' width=85%>

#### (i) Convolutional Neural Networks
- CNNs extract local relationships that are invariant across spatial dimensions.
- To adapt CNNs to time series datasets, convolutional filters designed to ensure only past information is used for forecasting.
- For an intermediate feature at hidden layer l, each causal convolutional filter takes the form below:
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;h^{l+1}_t=A((W*h)(l,t)),(W*h)(l,t)=\sum_{{\tau}=0}^kW(l,\tau)h^{l}_{t-{\tau}}" width=60%>
  ➔ h<sup>l</sup><sub>t</sub> ∈ **R**<sup>H<sub>in</sub></sup> is an intermediate state at layer l at time t
  ➔ * is the convolution operator, **W**(l, τ) ∈ **R**<sup>H<sub>out</sub>XH<sub>in</sub></sup> is a fixed filter weight at layer l
  ➔ A(.) is an activation function, representing any architecture-specific non-linear processing
  ➔ For CNNs that use a total of L convolutional layers, we note that the encoder ouptut is then z<sub>t</sub> = h<sup>L</sup><sub>t</sub>.
- Key implications for temporal relationships learnt by CNNs:
  - In line with the spatial invariance assumptions for standard CNNs, temporal CNNs assume that relationships are time-invariant
  ➔ using the same set of filter weights at each time step and across all time.
  - CNNs are only able to use inputs within its defined lookback window, or receptive field, to make forecasts. 
  🠊 the receptive field size k needs to be tuned carefully to ensure that the model can make use of all relevant historical information.
  
- **Dilated Convolutions**
➔ To alleviate computational difficulties where long-term dependencies are significant
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;(W*h)(l,t,d_l)=\sum_{{\tau}=0}^{\lfloor{k/d_l}\rfloor}W(l,{\tau})h^l_{t-d_l{\tau}}" width=40%>
➔ d<sub>l</sub> is a layer-specific dilation rate.
➔ Dilated convolutions can be interpreted as convolutions of a down-sampled version of the lower layer features,
reducing resolution to incorporate information from the distant past. 

#### (ii) Recurrent Neural Networks
- RNN cells contain an internal memory state which acts as a compact summary of past information. The memory state is recursively updated with new observations at each time step.
- Simplest RNN (Elman RNN):
  
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;y_{t+1}={\gamma}_y(W_yz_t+b_y),\\z_t={\gamma}_z(W_{z_1}z_{t-1}+W_{z_2}y_t+W_{z_3}x_t+W_{z_4}s+b_z)" width=60%>

➔ &gamma;<sub>y</sub>(.), &gamma;<sub>z</sub>(.) are network activation functions
- RNNs do not require the explicit specification of a lookback window as per the CNN case.
- Bayesian filters and RNNs are both similar in their maintenance of a hidden state which is recursively updated over time
  
- **Long Short-term Memory (LSTM)**
➔ Due to the infinite lookback window, older variants of RNNs can suffer from limitations in learning long-range dependencies in the data – due to issues with exploding and vanishing gradients
  <img src='https://user-images.githubusercontent.com/57218700/151647504-df5b0f8b-4644-4a36-80b8-e0a0400b397e.png' width=70%>
➔ Cell state c<sub>t</sub> stores long-term information, modulated through a series of gates
➔ z<sub>t-1</sub> is the hidden state, &sigma;(.) is the sigmoid activation function
➔ ☉ is the element-wise (Hadamard) product

#### (iii) Attention Mechanisms
- Improvements in long-term dependency learning with Transformer architectures achieving state-of-the-art performance in
multiple natural language processing applications
- Attention is a mechanism for a key-value lookup based on a given query:
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;h_t=\sum_{{\tau}=0}^k\alpha(\kappa_t,q_{\tau})v_{t-{\tau}}" width=25%>
  ➔ Where the key κ<sub>t</sub>, query q<sub>τ</sub> and value v<sub>t−τ</sub> are intermediate features produced at different time steps by lower levels of the network.
  ➔ &alpha;(κ<sub>t</sub>, query q<sub>τ</sub>) ∈ [0, 1] is the attention weight for t−τ generated at time t
  ➔ h<sub>t</sub> is the context vector output of the attention layer
- Attention aggregates features extracted by RNN encoders, with attention weights:
  
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;\alpha(t)=\text{softmax}(\eta_t),\\\eta_t=W_{\eta_1}tanh(W_{\eta_2}\kappa_{t-1}+W_{\eta_3}q_{\tau}+b_{\eta})" width=40%>
  ➔ &alpha;(t) = [&alpha;(t,0),...,&alpha;(t,k)] is a vector of attention weights
  ➔ κ<sub>t-1</sub>, query q<sub>τ</sub> are outputs from LSTM encoders used for feature extraction
- Attention provides two key benefits:
  - Networks with attention areable to directly attend to any significant events that occur. 
  - Attention-based networks can also learn regime-specific temporal dynamics by using distinct attention weight patterns for each regime.

#### (iv) Outputs and Loss Functions
- By customising of decoder and output layer of the neural network to match the desired target type
  ➔ discrete and continuous targets
  ➔ Regardless of the form of the target, predictions can be further divided into point estimates and probabilistic forecasts.
  - **Point estimates** 
  ➔ This essentially involves reformulating the problem to a classification task for discrete outputs and regression task for continuous outputs
  ➔ For one-step-ahead forecasts of binary and continuous targets, networks are trained using binary cross-entropy and mean square error loss functions respectively:

  - **Probabilistic Outputs**
    ➔ Understanding the uncertainty of a model’s forecast can be useful for decision makers in different domains
    ➔ A common way to model uncertainties is to use deep neural networks to generate parameters of known distributions. 
    ➔ For example, Gaussian distributions are typically used for forecasting problems with continuous targets, with the networks outputting means and variance parameters for the predictive distributions at each step:
    
    <img src="https://latex.codecogs.com/svg.latex?\Large&space;y_{t+{\tau}}{\sim}N(\mu(t,\tau),\zeta(t,\tau)^2),\\\mu(t,\tau)=W_{\mu}h^L_t+b_{\mu},\,\,\zeta(t,\tau)=\text{softplus}(W_{\sum}h^L_t+b_{\sum})." width=60%>  
### (b) Multi-horizon Forecasting Models
- It is often beneficial to have access to predictive estimates at multiple points in the future allowing decision makers to visualise trends over a future horizon, and optimise their actions across the entire path

  <img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{y_{t+\tau}}=f(y_{t-k:t},x_{t-k:t},u_{t-k:t+\tau},s,\tau)" width=45%>  
  ➔ τ ∈ {1, ..., τ<sub>max</sub>} is a discrete forecast horizon, u<sub>t</sub> are known future inputs across the entire horizon
    <img src='https://user-images.githubusercontent.com/57218700/151648594-363a350e-acbc-40e3-a8b4-8cb2fdc4066b.png' width=80%>

#### (i) Iterative Methods
- Autoregressive deep learning architectures producing multi-horizon forecasts by recursively feeding samples of the target into future time steps 
- As autoregressive models are trained in the exact same fashion as one-step-ahead prediction models via BPTT, the iterative approach allows for the easy generalisation of standard models to multi-step forecasting
- Limitations : large error accumulations, all inputs but the target are known at run-time

#### (ii) Direct Methods
- Direct methods alleviate the issues with iterative methods by producing forecasts directly using all available inputs.
- Sequence-to-sequence architectures, using an encoder to summarise past information and a decoder to combine them with known future inputs

## Incorporating domain knowledge with hybrid models
- Underperformance machine learning methods:
  - The flexibility of machine learning methods can be a double-edged sword making them prone to overfitting
  ➔ simpler models may potentially do better in low data regimes
  -  Similar to stationarity requirements of statistical models, machine learning models can be sensitive to how inputs are pre-processed, which ensure that data distributions at training and test time are similar.
- A recent trend in deep learning has been in developing hybrid models which address these limitations, demonstrating improved performance over pure statistical or machine learning models.
➔ Hybrid models allow domain experts to inform neural network training using prior information reducing the hypothesis space of the network and improving generalisation.
➔ Hybrid models allow for the separation of stationary and non-stationary components, and avoid the need for custom input pre-processing. (`ES-RNN` uses exponential smoothing to capture non-stationary trends and learns additional effects with the RNN.)
- Hybrid models utilise deep neural networks in two manners: 
  - to encode time-varying parameters for non-probabilistic parametric models
  - to produce parameters of distributions used by probabilistic models 

### (a) Non-probabilisic hybrid model
- Non-probabilistic hybrid models modify these forecasting equations to combine statistical and deep learning components. 
- The `ES-RNN` utilises the update equations of the Holt-Winters exponential smoothing model combining multiplicative level and seasonality components with deep learning outputs:
  
    <img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{y}_{i,t+\tau}=\text{exp}(W_\text{ES}h^L_{i,t+\tau}+b_\text{ES}){\times}l_{i,t}\times\gamma_{i,t+\tau},\\l_{i,t}=\beta^{(i)}_1y_{i,t}/\gamma_{i,t}+(1-\beta^{(i)}_1)l_{i,t-1},\\\gamma_{i,t}=\beta^{(i)}_2y_{i,t}/l_{i,t}+(1-\beta^{(i)}_2)\gamma_{i,t-\kappa}" width=48%>  
  ➔ h<sup>L</sup><sub>i,t+τ</sub> is the final layer of the network for the τth-step-ahead forecast
  ➔ l<sub>i,t</sub> is a level component, &gamma;<sub>i,t</sub> is a seasonality component with period κ.
  ➔ &beta;<sup>(i)</sup><sub>1</sub>, &beta;<sup>(i)</sup><sub>2</sub> are entity-specific static coefficients.

### (b) Probabilistic hybrid models
- Where distribution modelling is important utilising probabilistic generative models for temporal dynamics such as Gaussian
processes and linear state space models.
- Deep State Space Models encode time-varying parameters for linear state space models as below performing inference via the Kalman filtering equations:
  
    <img src="https://latex.codecogs.com/svg.latex?\Large&space;y_t=a(h^L_{i,t+\tau})^{T}l_{t}+\phi(h^L_{i,t+\tau})\epsilon_t,\\l_t=F(h^L_{i,t+\tau})l_{t-1}+q(h^L_{i,t+\tau})+\Sigma(h^L_{i,t+\tau}){\odot}{\Sigma}_t" width=48%>  
  ➔ l<sub>t</sub> is the hidden latent state, a(.), F(.), q(.) are linear transformations of h<sup>L</sup><sub>i,t+τ</sub>
  ➔ Φ(.), Σ(.) are linear transformations with softmax activations
  ➔ &epsilon;<sub>t</sub> is a univariate residual and Σ<sub>t</sub>~N(0,I) is a multivariate normal random variable.

## Facilitating decision support using deep neural networks
- Model builders are mainly concerned with the accuracy of their forecasts, end-users typically use predictions to guide their future actions.
- While time series forecasting is a crucial preliminary step, a better understanding of both temporal dynamics and the motivations behind a model’s forecast can help users further optimise their actions. 
  ➔ interpretability and causal inference

### (a) Interpretability with time series data
- Need to understand both how and why a model makes a certain prediction
- End-users can have little prior knowledge with regards to the relationships present in their data, with datasets growing in size and complexity in recent times. 
- **Techniques for Post-hoc Interpretability**
  ➔ Help to identify important features or examples without modifying the original weights.
    - One possible approach is to apply simpler interpretable surrogate models between the inputs and outputs of the neural network. ➔ LIME, SHAP
    - Gradient-based method analyse network gradients to determine which input features have the greatest impact on loss functions. ➔ saliency maps,  influence functions
- **Inherent Interpretability with Attention Weights**
➔ Directly design architectures with explainable components, typically in the form of strategically placed attention layers
➔ An analysis of attention weights can then be used to understand the relative importance of features at each time step.

### (b) Counterfactual Predictions & Causal Inference Over Time
- Counterfactual predictions are particularly useful for scenario analysis applications allowing users to evaluate how different sets of actions can impact target trajectories.
  - Useful from a historical angle : determining what would have happened if a different set of circumstances had occurred
  - Useful from a forecasting perspective : determining which actions to take to optimise future outcomes.
- While a large class of deep learning methods exists for estimating causal effects in static settings, the key challenge in time series datasets is the presence of time-dependent confounding effects.
  - Methods to train deep neural networks while adjusting for timedependent confounding, based on extensions of statistical techniques and the design of new loss functions.
  ➔ IPTW, G-computation framework, new loss functions

## Conclusions and future directions
- Deep neural networks typically require time series to be discretised at regular intervals, making it difficult to forecast datasets where observations can be missing or arrive at random intervals
- Time series often have a hierarchical structure with logical groupings between trajectories
➔ Retail forecasting, where product sales in the same geography can be affected by common trends
- The development of architectures which explicit account for such hierarchies could be an interesting research direction,
and potentially improve forecasting performance over existing univariate or multivariate models.