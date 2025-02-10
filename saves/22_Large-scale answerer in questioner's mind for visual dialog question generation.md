## Large-scale answerer in questioner's mind for visual dialog question generation
- Authors : Lee, Sang-Woo and Gao, Tong and Yang, Sohee and Yoo, Jaejun and Ha, Jung-Woo
- Journal : arXiv preprint
- Year : 2019
- Link : https://arxiv.org/pdf/1902.08355.pdf



### Abstract
- `Answerer in Questioner's Mind (AQM)`
  - `AQM` benefits from asking a question that would maximize the information gain when it is asked.
➔ Due to its intrinsic nature of explicitly calculating the information gain, `AQM` has a limitation when the solution space is very large.
- We propose `AQM+` that can deal with a large-scale problem and ask a question that is more coherent to the current context of the dialog.
- We evaluate our method on GuessWhich and the proposed `AQM+` reduces more than 60% of error as the dialog proceeds, while the comparative algorithms diminish the error by less than 6%.


### Introduction
-  AQM benefits from explicitly calculating the posterior distribution and finding a solution analytically. The authors showed promising results in the task-oriented dialog problem, such as **GuessWhat**, where a questioner tries to find an object that is in answerer’s mind via a series of Yes/No questions. 
- The candidates are confined to the objects that are presented in the given image (less than ten on average). However,
this simplified task may not be general enough to practical problems where the number of objects, questions and answers are typically unrestricted.
- Because the computational complexity vastly increases to explicitly calculate the information gain over the size of the entire search space, the original `AQM` algorithm is not scalable to a large scale problem. 
- Retrieval-based models, which are basically discriminative models that select a response from a predefined candidate set of system responses, are critical not to generate sentences that are ill-structured or irrelevant to the task. 
  - Such a discriminative approach does not fit well with complicated task-oriented visual dialog tasks, because asking an appropriate question considering the visual context is crucial to successfully tackle the problem. 
- We propose `AQM+` that can handle a more complicated problem where the number of candidate classes is extremely large.
  - At every turn, `AQM+` generates a question considering the context of the previous dialog, which is desirable in practice.
  - `AQM+` generates candidate questions and answers at every turn to ask an appropriate question in the context.

### Related Works
- **GuessWhat**
  - Task-oriented dialog tasks, where the goal is to figure out a target object in the image through a dialog that the answerer has in mind.
  - The answer form of yes or no ➔ Easy task
- **GuessWhich**
  -  A cooperative two-player game that one player tries to figure out an image out of 9,628 that another has in mind.
  -  Using Visual Dialog dataset which includes human dialogs on MSCOCO images as well as the captions that are generated.


### Algorithm: AQM+
#### Problem Setting
- At each turn t, Qbot generates an appropriate question q<sub>t</sub> and guesses the target class c given a previous history of the dialog $h_{t-1} = (q_{1:t−1}, a_{1:t-1}, h_0)$.
  - a<sub>t</sub> is the t-th answer and h<sub>0</sub> is an initial context that can be obtained before the start of the dialog. 
  - We refer to the random variables of target class and the t-th answer as $C$ and $A_t$, respectively. Note that the t-th question is not a random variable in our information gain calculation. 
  - To distinguish from the random variables, we use a bold face for a set notation of target class, question, and answers; i.e. $C$, $Q$, and $A$.
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/a8c4567b-429b-4947-851b-7a36e879f885' width=80%>


#### Preliminary: SL, RL, and `AQM` Approches
- Qbot consists of two RNN modules: 
  - Qgen: a question generator finding the solution that maximizes its distribution:
$$q_t ^† = \text{argmax} \,\, p^† (q_t | h_{t-1} ).$$
  - Qscore: a class guesser using score function for each class $f^‡ (c|h_t)$.
- These two RNN-based models are substituted to the calculation that explicitly finds and analyic solution.
  - It finds a question that maximizes information gain or mutual information $\tilde{I}$.
  - <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/4b0de934-b02d-4c4c-a0cb-9b75ae392b9a' width=35%>
  - <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/714b60fc-5327-4f5b-95ec-33bd897c74a5' width=65%>
  - A posterior function $\hat{p}$ can be calculated with a following equation in a sequential way, where $\hat{p}'$ is a prior function given $h_0$.
  - <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/97bf2b6a-0c76-4f9c-a16a-62c3af9556d7' width=65%>
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/2837588d-2a41-45fa-a7c8-293b70b27ef9' width=80%>
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/4aaf1438-8e34-4dc6-995d-5de5516a6e73' width=65%>
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/005cadd6-dceb-4816-8e5a-18ce462f273c' width=65%>
- As `AQM` uses full set of $C$ and $A$, the complexity depends on the size of $C$ and $A$. 
- For the question selection, `AQM` uses a predefined set of candidate questions $(Q_{fix})$, which is not changed for a
different turn.


#### `AQM+` Algorithm
- `AQM+` uses sampling-based approximation, for tackling the large-scale task-oriented dialog problem. 
- The core differences of `AQM+` from the previous `AQM`
  - The candidate question set $Q_{t,gen}$ is sampled from $p^†(q_t | h_{t-1})$ using a beam search at every turn.
  - The answer model $\tilde{p}$ that Qbot has in mind is not a binary classifier (yes/no) but an RNN generator.
    - AprxAgen $\tilde{p}$ is not even an appropriate assumption when the previous and current questions are sequentially related.
$\tilde{p} (a_t | c, q_t) \neq \tilde{p} (a_t | c, q_t, h_{t-1})$
  - To approximate the information gain of each question, the subsets of $A$ and $C$ are also sampled at every turn.
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/14838f8f-18f1-4d91-bd89-70b53b40e35d' width=75%>
    - $C_{t, topk}$: top-K posterior test images from $\hat{p}(c | h_{t-1})$
    - $Q_{t, gen}$: top-K likelihood questions using the beam search from $p^† (q_t | h_{t-1})$
    - $A_{t, topk} (q_t)$: top-1 generated answers from AprxAgen for each question $q_t$ and each class in $C_{t, topk}$ from $\tilde{p} (a_t | c, q_t, h_{t-1})$


#### Learning
- In SL approach, Qgen and Qscore are trained from the training data, which have the same or similar distribution to that of the training data used in training Abot. 
  - In indA setting of `AQM` approach, aprxAgen is trained from the training data. 
- In RL approach, Qbot uses dialogs made by the conversation of Qbot and Abot and the result of the game as the objective
function (i.e. reward). 
  - In depA setting of `AQM` approach, aprxAgen is trained from the questions in the training data and following answers obtained in the conversation between Qbot and Abot. 
- We also use the term trueA, referring to the setting where aprxAgen is the same as Agen, i.e. they share the same parameters. 


### Experiments
#### Experimental Setting and Comparative Results
- GuessWhich is to figure out a correct answer out of 9,628 test images by asking a sequence of questions.
- We use both non-delta setting and delta setting to test the performance of `AQM+`.
- Our model uses five modules, Qgen, Qscore, aprxAgen, Qinfo, and Qpost.
  - We set $|C_{t,topk}| = |Q_{t,gen}| = |A_{t, topk} (qt)| = 20$. 
  - The epoch for SL-Q is 60. The epoch for RL-Q and RL-QA is 20 for non-delta, and 15 for delta, respectively.
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/f648aeb4-6adb-4f63-80b7-19d7cfb46921' width=70%>


#### Ablation Study
- No caption experiment
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/daf888b6-d818-4ed7-a64f-c177a8440172' width=70%>
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/18a286f9-aade-40cb-9115-93eced3d98d0' width=70%>

- Random candidate answers experiment
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/f007fb83-39da-448c-8ff5-3f2459e19ef0' width=70%>


### Conclusion
-  `AQM+` can ask an appropriate question considering the context of the dialog, handle the responses in a sentence form, and efficiently estimate information gain of the target class with a given question. 
- `AQM+` not only outperforms the comparative SL and RL algorithms, but also enlarges the gap between `AQM+` and the comparative algorithms comparing to the performance gaps reported in GuessWhat. 


