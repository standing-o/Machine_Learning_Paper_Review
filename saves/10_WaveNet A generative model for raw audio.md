## WaveNet: A generative model for raw audio
- Authors : Van Den Oord, Aäron, et al.
- Journal : SSW
- Year : 2016
- Link : https://arxiv.org/abs/1609.03499

### Abstract
- ``WaveNet``, a deep neural network for generating raw audio waveforms. The model is fully probabilistic and autoregressive, with the predictive distribution for each audio sample conditioned on all previous ones.
- A single ``WaveNet`` can capture the characteristics of many different speakers with equal fidelity, and can switch between them by conditioning on the speaker identity.

### Introduction
- Modeling joint probabilities over pixels or words using neural architectures as products of conditional distributions yields state-of-the-art generation.
- ``WaveNet`` : an audio generative model based on the PixelCNN architecture. The main contributions of this work are as follows:
1. ``WaveNets`` can generate raw speech signals with subjective naturalness never before reported in the field of text-to-speech (TTS)
2. In order to deal with long-range temporal dependencies, we develop new architectures based on dilated causal convolutions, which exhibit very large receptive fields
3. A single model can be used to generate different voices, conditioned on a speaker identity
4. Promising when used to generate other audio modalities such as music.

### ``Wavenet``
- The joint probability of a waveform x = {x<sub>1</sub>, . . . , x<sub>T</sub>} is factorised as a product of conditional probabilities as follows:
  <img src='https://user-images.githubusercontent.com/57218700/156532886-38fd7423-52e8-43df-9ead-48c84a0f8712.png' width=35%>
  ➔ Each audio sample x<sub>t</sub> is therefore conditioned on the samples at all previous timesteps.
-  The conditional probability distribution is modelled by a stack of convolutional layers. No pooling layers.
- Output of the model has the same time dimensionality as the input. The model outputs a categorical distribution over the next value x<sub>t</sub> with a softmax layer.
- The model is optimized to maximize the log-likelihood of the data w.r.t. the parameters.
➔ Log-likelihoods are tractable, we tune hyper-parameters on a validation set and can easily measure overfitting/underfitting.

#### Dilated Causal Convolutions
- Dilated covolution layer + causal convolution layer
**1. Causal convolution layer**
  <img src='https://user-images.githubusercontent.com/57218700/156533547-a0c86063-fa29-45b7-a3db-063ce8dd6f4c.png' width=70%>
➔ By using causal convolutions, we make sure the model cannot violate the ordering in which we model the data
 the prediction
➔ p(x<sub>t+1</sub> | x<sub>1</sub>, ..., x<sub>t</sub>) emitted by the model at timestep t cannot depend on any of the future timesteps x<sub>t+1</sub>, x<sub>t+2</sub>, . . . , x<sub>T</sub>.
➔ Constructing a mask tensor and multiplying this elementwise with the convolution kernel before applying it.
➔ At training time, the conditional predictions for all timesteps can be made in parallel because all timesteps of ground truth x are known.
➔ Faster to train than RNNs, but they require many layers, or large filters to increase the receptive field 🠊 **Dilated convolution**
**2. Dilated causal convolution**
  <img src='https://user-images.githubusercontent.com/57218700/156534439-789c3e5f-825a-4b44-a296-be30a4c58649.png' width=70%>
➔ Dilated convolution is a convolution where the filter is applied over an area larger than its length by skipping input values with a certain step. It is equivalent to a convolution with a larger filter derived from the original filter by dilating it with zeros, but is significantly more efficient.
➔ In this paper, the dilation is doubled for every layer up to a certain point (512) and then repeated (1,2,4,...,512,1,2,4,...,512).

#### Softmax Distribution
- Modeling the conditional distributions p (x<sub>t</sub> | x<sub>1</sub>, . . . , x<sub>t−1</sub>) over the individual audio samples. 
- Categorical distribution is more flexible and can more easily model arbitrary distributions because it makes no assumptions about their shape.
➔ Because raw audio is typically stored as a sequence of 16-bit integer values (one per timestep), a softmax layer would need to output 65,536 probabilities per timestep to model all possible values. 
➔ To make this more tractable, we first apply a **µ-law companding transformation** to the data, and then quantize it to 256 possible values:
  <img src='https://user-images.githubusercontent.com/57218700/156535111-a132087e-2679-4dc5-9aa7-b041ca5a44c9.png' width=35%>
where -1<x<sub>t</sub><1 and &mu;=255.
➔ This non-linear quantization produces a significantly better reconstruction than a simple linear quantization scheme.

#### Gated activation units
<img src='https://user-images.githubusercontent.com/57218700/156535657-df64dd95-6103-4b3e-bc83-d6c5fa2eb6da.png' width=40%>
where * denotes a convolution operator, &sigma; is a sigmoid function, k is the layer index, f and g denote filter and gate, respectively.

#### Residual and skip connections
- Residual block and the entire architecture
  <img src='https://user-images.githubusercontent.com/57218700/156535940-08bce9d7-96ea-4fca-a79e-bfb211d2f831.png' width=70%>
  ➔ Both residual and parameterised skip connections are used throughout the network, to speed up convergence and enable training of much deeper models.

#### Conditional ``Wavenet``
- Given an additional input h, ``WaveNets`` can model the conditional distribution p(x | h) of the audio given this input. 
  <img src='https://user-images.githubusercontent.com/57218700/156536370-f4dce27e-4a7f-4358-9d99-2d8db32bc169.png' width=40%>
  ➔ By conditioning the model on other input variables, we can guide ``WaveNet``’s generation to produce audio with the required characteristics.
  ➔  In a multi-speaker setting we can choose the speaker by feeding the speaker identity to the model as an extra input. For TTS we need to feed information about the text as an extra input.
- We condition the model on other inputs in two different ways:
1. **Global conditioning** is characterised by a single latent representation h that influences the output distribution across all timesteps, e.g. a speaker embedding in a TTS model.
  <img src='https://user-images.githubusercontent.com/57218700/156536660-0315514f-709e-4f36-8f75-1bd4b3c23ce3.png' width=55%>
  where V<sub>*,k</sub> is a learnable linear projection, and the vector V<sup>T</sup><sub>*,k</sub> is broadcast over the time dimension.
  
2. For **local conditioning** we have a second timeseries ht, possibly with a lower sampling frequency than the audio signal, e.g. linguistic features in a TTS model.
➔ We first transform this time series using a transposed convolutional network (learned upsampling) that maps it to a new time series
y = f(h) with the same resolution as the audio signal, which is then used in the activation unit as follows:
<img src='https://user-images.githubusercontent.com/57218700/156537001-2ec644cc-9fb8-4e22-829c-940e5b2841fb.png' width=55%>
where V<sub>f,k</sub>*y is now a 1 X 1 convolution.

### Experiments
#### Multi-Speaker speech generation
- VCTK (Voice cloning toolkit) dataset
- The conditioning was applied by feeding the speaker ID to the model in the form of a one-hot vector.
- A single ``WaveNet`` was able to model speech from any of the speakers by conditioning it on a onehot encoding of a speaker.
- We observed that the model also picked up on other characteristics in the audio apart from the voice itself.

#### Text-to-speech
- ``WaveNets`` for the TTS task were locally conditioned on linguistic features which were derived from input texts. We also trained WaveNets conditioned on the logarithmic fundamental frequency (log F<sub>0</sub>) values in addition to the linguistic features.
-  Hidden Markov model (HMM), long short-term memory recurrent neural network (LSTM-RNN)
- ``Wavenet``(L+F) : Linguistic features + logF<sub>0</sub>
-  The performance of ``WaveNets`` for the TTS task:
1. Subjective paired comparison tests
➔ After listening to each pair of samples, the subjects were asked to choose which they preferred
2. Mean opinion score (MOS) tests
➔ After listening to each stimulus, the subjects were asked to rate the naturalness of the stimulus in a five-point Likert scale score
- ``WaveNet`` improved the previous state of the art significantly, reducing the gap between natural speech and best previous model by more than 50%.
  <img src='https://user-images.githubusercontent.com/57218700/156538279-825698ea-6081-433b-9da8-1d7ed9cb4e07.png' width=70%>

### Conclusion
- ``WaveNets`` are autoregressive and combine causal filters with dilated convolutions to allow their receptive fields to grow exponentially with depth, which is important to model the long-range temporal dependencies in audio signals.
- ``WaveNets`` can be conditioned on other inputs in a global (e.g. speaker identity) or local way (e.g. linguistic features).