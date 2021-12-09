# Machine-learning Paper review
- Machine learning Paper review and code implementation
- The summary of papers are recorded in Issues (inspired by kweonwooj's [Github](https://github.com/kweonwooj/papers/issues) :thumbsup:)
- Presentation slides are made for biweekly seminar in SClab & Data analysis club
- Mainly study on machine learning optimization and computer vision.
- Aug. 19, 2020 ~ Present


## 👉 Table of Contents
- [Optimization](#optimization)
- [Computer Vision](#computer-vision)
- [Classic Papers (before 2012)](#classic-papers)
-----------------------
## :chart_with_downwards_trend: Optimization
#### 1. [Momentum] On the importance of initialization and momentum in deep learning. | [`[sutskever2013.pdf]`](http://proceedings.mlr.press/v28/sutskever13.pdf)
| [Presentation](https://github.com/OH-Seoyoung/Machine-learning_Paper_review/blob/master/Optimization/1_Initialization_and_Momentum/20210813_Initialization_and_Momentum.pdf) |

#### 2. [Adam] Adam: A method for stochastic optimization. | [`[kingma2014.pdf]`](https://arxiv.org/pdf/1412.6980.pdf?source=post_page---------------------------)
| [Presentation](https://github.com/OH-Seoyoung/Machine-learning_Paper_review/blob/master/Optimization/2_ADAM/20210826_Adaptive_moment_estimation.pdf) |

#### 3. [Dropout] Dropout: a simple way to prevent neural networks from overfitting. | [`[srivastava2014.pdf]`](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_campaign=buffer&utm_content=buffer79b43&utm_medium=social&utm_source=twitter.com)
| [Presentation](https://github.com/OH-Seoyoung/Machine-learning_Paper_review/blob/master/Optimization/3_Dropout/20210907_Dropout.pdf) |   

#### 4. [Batch normalization] Batch normalization: Accelerating deep network training by reducing internal covariate shift. | [`[ioffe2015.pdf]`](http://proceedings.mlr.press/v37/ioffe15.pdf)
| [Presentation](https://github.com/OH-Seoyoung/Machine-learning_Paper_review/blob/master/Optimization/4_Batch_normalization/20211005_Batch_normalization.pdf) |  

#### 5. [HighwayNet] Training very deep networks. | [`[srivastava2015.pdf]`](https://arxiv.org/pdf/1507.06228.pdf)
| [Presentation](https://github.com/OH-Seoyoung/Machine-learning_Paper_review/blob/master/Optimization/5_HighwayNet/20211019_HighwayNet_Training_very_deep_networks.pdf) |

#### 6. [He initialization] Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. | [`[he2015.pdf]`](https://openaccess.thecvf.com/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)
| [Presentation](https://github.com/OH-Seoyoung/Machine-learning_Paper_review/blob/master/Optimization/6_He_initialization/20211102_He_initialization.pdf) |  

<a href='#table-of-contents'></a>
<br/>
  
## :tv: Computer Vision
#### 1. [LeNet] Gradient-based learning applied to document recognition. | [`[lecun1998.pdf]`](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=726791&casa_token=ElGW6XRIra8AAAAA:UDZPHfQO58TTOxZo5Kw-gSpmwo9t7DWe4u197dJuKNUwJ-ZI1TomItrS-7PL0eqnnNXKalMY_Q)
| [Presentation](https://github.com/OH-Seoyoung/Machine-learning_Paper_review/blob/master/Computer_vision/1_Lenet/20201201_Lenet.pdf) | [Code](https://github.com/OH-Seoyoung/Machine-learning_Paper_review/blob/master/Computer_vision/1_Lenet/Simple_implementation_of_CNN.ipynb) |

<a href='#table-of-contents'></a>
<br/>
  
## :heavy_check_mark: Classic papers
#### 1. [Turing Machine] On computable numbers, with an application to the Entscheidungsproblem. | [`[turing1936.pdf]`](https://www.wolframscience.com/prizes/tm23/images/Turing.pdf)
| [Presentation](https://github.com/OH-Seoyoung/Machine-learning_Paper_review/blob/master/Classic_papers/1_Turing_Machine/20200929_Turing_Machine.pdf) | 

#### 2. [Imitation Game] Computing machinery and intelligence. | [`[turing2009.pdf]`](http://www.cse.chalmers.se/~aikmitr/papers/Turing.pdf#page=442)
| [Presentation](https://github.com/OH-Seoyoung/Machine-learning_Paper_review/blob/master/Classic_papers/2_Imitation_Game/20201006_Imitation_game.pdf) |

#### 3. [Back-propagation] Learning representations by back-propagating errors. | [`[hinton1986.pdf]`](http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)
| [Presentation](https://github.com/OH-Seoyoung/Machine-learning_Paper_review/blob/master/Classic_papers/3_Back-Propagation/20201110_Back-Propagation.pdf) | [Code](https://github.com/OH-Seoyoung/Machine-learning_Paper_review/blob/master/Classic_papers/3_Back-Propagation/Simple_implementation_of_back-propagation.ipynb) |

#### 4. [Deep belief net] Reducing the dimensionality of data with neural networks. | [`[hinton2006.pdf]`](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.3788&rep=rep1&type=pdf)
| [Presentation](https://github.com/OH-Seoyoung/Machine-learning_Paper_review/blob/master/Classic_papers/4_Dimensionality_Reduction_DBN/20210121_Dimensionality_Reduction_DBN.pdf) | [Code1](https://github.com/OH-Seoyoung/Machine-learning_Paper_review/blob/master/Classic_papers/4_Dimensionality_Reduction_DBN/AE_and_PCA/Multi-layer_Autoencoder_and_PCA.ipynb), [Code2](https://github.com/OH-Seoyoung/Machine-learning_Paper_review/tree/master/Classic_papers/4_Dimensionality_Reduction_DBN/RBM_and_PCA_with_MNIST) |

- In addition, we conducted a simple study that applied dimensionality reduction using PCA, RBM to classification problems.  
-->> ["Dimensionality reduction methods and Deep learning approach"](https://github.com/OH-Seoyoung/Machine-learning_Paper_review/blob/master/Classic_papers/4_Dimensionality_Reduction_DBN/RBM_and_PCA_with_MNIST/Poster.pdf)

#### 5. [Unsupervised Pretraining] Why does unsupervised pre-training help deep learning? | [`[erhan2010.pdf]`](http://proceedings.mlr.press/v9/erhan10a/erhan10a.pdf) 
| [Presentation](https://github.com/OH-Seoyoung/Machine-learning_Paper_review/blob/master/Classic_papers/5_Unsupervised_Pre-training/20210204_Unsupervised_Pre-training.pdf) | 

<a href='#table-of-contents'></a>
