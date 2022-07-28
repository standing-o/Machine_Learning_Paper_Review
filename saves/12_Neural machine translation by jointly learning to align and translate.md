## Neural machine translation by jointly learning to align and translate
- Authors : Bahdanau, Dzmitry and Cho, Kyunghyun and Bengio, Yoshua
- Journal : arXiv
- Year : 2014
- Link : https://arxiv.org/pdf/1409.0473.pdf

### Abstract
- `Neural machine translation (NMT)` often belong to a family of encoder–decoders and encode a source sentence into a fixed-length vector from which a decoder generates a translation.
-  The use of a fixed-length vector is a bottleneck in improving the performance of this basic encoder–decoder architecture, and propose to extend this by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word, without having to form these parts as a hard segment explicitly.

### Introduction
- The performance of a basic encoder–decoder deteriorates rapidly as the length of an input sentence increases.
- We introduce an extension to the encoder–decoder model which learns to `align` and translate jointly. 
- Each time the proposed model generates a word in a translation, it (soft-)searches for a set of positions in a source sentence where the most relevant information is concentrated. 
- The model then predicts a target word based on the context vectors associated with these source positions and all the previous generated target words.
- It encodes the input sentence into a sequence of vectors and chooses a subset of these vectors adaptively while decoding the translation. 
➔ This frees a neural translation model from having to squash all the information of a source sentence, regardless of its length, into a fixed-length vector.

### RNN Encoder-decoder
- In the Encoder–Decoder framework, an encoder reads the input sentence, a sequence of vectors x = (x<sub>1</sub>, · · · , x<sub>Tx</sub>), into a vector c.
- The most common approach is to use an RNN s.t. h<sub>t</sub> = f(x<sub>t</sub>, h<sub>t-1</sub>) and c = q({h<sub>1</sub>, ..., h<sub>Tx</sub>}), where h<sub>t</sub> ∈ R<sup>n</sup> is a hidden state at time t, and c is a vector generated from the sequence of the hidden states. 
- f and q are some nonlinear functions. For instances, Sutskever et al. (2014) used an LSTM as f and q({h<sub>1</sub>, ..., h<sub>Tx</sub>}) = h<sup>T</sup>.
- The decoder is often trained to predict the next word y<sub>t'</sub> given the context vector c and all the previously predicted words {y<sub>1</sub>, ..., y<sub>T</sub>}.
➔ i.e., the decoder defines a probability over the translation y by decomposing the joint probability into the ordered conditionals:
  <img src='https://user-images.githubusercontent.com/57218700/166451790-6b395ebd-9d00-489d-8cd7-53728368f7f6.png' width=35%>
where y = y<sub>1</sub>, ..., y<sub>Ty</sub>.
- With an RNN, each conditional probability is modeled as p(yt | {y<sub>1</sub>, ..., y<sub>t-1</sub>} , c) = g(y<sub>t−1</sub>, s<sub>t</sub>, c), where g is a nonlinear, potentially multi-layered, function that outputs the probability of y<sub>t</sub>, and s<sub>t</sub> is the hidden state of the RNN.

###  Learning to `align` and translate (New model architecture)
#### Decoder : General description
- Each conditional probability:
  <img src='https://user-images.githubusercontent.com/57218700/166452117-5f645d40-ee8a-4e45-bd73-f79c838cb07c.png' width=35%>
  where s<sub>i</sub> is an RNN hidden state for time i, computed by s<sub>i</sub> = f(s<sub>i-1</sub>, y<sub>i-1</sub>, c<sub>i</sub>).
  ➔ The probability is conditioned on a distinct context vector ci for each target word y<sub>i</sub>.
- The context vector ci depends on a sequence of annotations (h<sub>1</sub>, ..., h<sub>Tx</sub>) to which an encoder maps the input sentence. 
-  Each annotation h<sub>i</sub> contains information about the whole input sequence with a strong focus on the parts surrounding the i-th word of the input sequence.
- The context vector c<sub>i</sub> is computed as a weighted sum of these annotations h<sub>i</sub>:
  <img src='https://user-images.githubusercontent.com/57218700/166452457-4ca25e04-3110-43e2-9213-0451c9232d9e.png' width=20%>
- The weight α<sub>ij</sub> of each annotation h<sub>j</sub> is computed by:
  <img src='https://user-images.githubusercontent.com/57218700/166452599-6c6b3c4d-aabb-4e7a-8ed7-2b9cc937c7c8.png' width=50%>
  is an `alignment` model which scores how well the inputs around position j and the output at position i match. 
  ➔ The score is based on the RNN hidden state s<sub>i−1</sub> and the j-th annotation hj of the input sentence.
- We parametrize the `alignment` model a as a feedforward neural network which is jointly trained with
all the other components of the proposed system.
- The `alignment` model directly computes a soft `alignment`, which allows the gradient of the cost function to be backpropagated through. This gradient can be used to train the `alignment` model as well as the whole translation model jointly.

#### Encoder : Bidirectional RNN for annotating sequences
- We would like the annotation of each word to summarize not only the preceding words, but also the following words.
➔ We propose to use a bidirectional RNN (BiRNN, Schuster and Paliwal, 1997).
- We obtain an annotation for each word x<sub>j</sub> by concatenating the forward hidden state
  h<sub>j</sub><sup>-></sup> and the backward one h<sub>j</sub><sup><-</sup>:
  <img src='https://user-images.githubusercontent.com/57218700/166454190-c7a0ef5d-0335-46d2-9667-447efab143bd.png' width=20%>
- The annotation hj contains the summaries of both the preceding words and the following words. Due to the tendency of RNNs to better represent recent inputs, the annotation h<sub>j</sub> will be focused on the words around x<sub>j</sub>.
➔  This sequence of annotations is used by the decoder and the `alignment` model later to compute the context vector

### Experiment settings
- **Dataset** : WMT ’14 contains the English-French parallel corpora
➔ After a usual tokenization, we use a shortlist of 30,000 most frequent words in each language to train our models. Any word not included in the shortlist is mapped to a special token ([UNK]).
- **Models** : RNN Encoder–Decoder (RNNencdec, Cho et al.), **RNNsearch**(proposed model), 
➔ We train each model twice: first with the sentences of length up to 30 words (RNNencdec-30, RNNsearch-30) and then with the sentences of length up to 50 word (RNNencdec-50, RNNsearch-50).
- SGD with Adadelta : Each SGD update direction is computed using a minibatch of 80 sentences. 
- Once a model is trained, we use a beam search to find a translation that approximately maximizes the conditional probability.

### Results
- BLEU score Table :  the performance of the RNNsearch is as high as that of the conventional phrase-based translation system (Moses)
  <img src='https://user-images.githubusercontent.com/57218700/166454973-186617d8-3534-4c56-9a4e-c34fcb1a5d71.png' width=40%>
- The performance of RNNencdec dramatically drops as the length of the sentences increases.
- Both RNNsearch-30 and RNNsearch-50 are more robust to the length of the sentences. RNNsearch50, especially, shows no performance deterioration even with sentences of length 50 or more.
  <img src='https://user-images.githubusercontent.com/57218700/166455436-d6dfef84-932d-4a32-8e1f-b6d702d412fe.png' width=55%>

### Conclusion
- We extended the basic encoder–decoder by letting a model (soft-)search for a set of input words, or their annotations computed by an encoder, when generating each target word. 
→ This frees the model from having to encode a whole source sentence into a fixed-length vector, and also lets the model focus only on information relevant to the generation of the next target word. 
➔ This has a major positive impact on the ability of the neural machine translation system to yield good results on longer sentences.
- All of the pieces of the translation system, including the `alignment` mechanism, are jointly trained towards a better log-probability of producing correct translations.