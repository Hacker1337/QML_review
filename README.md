# Quantum Machine learning in NISQ era
<img src="https://github.com/Hacker1337/QML_review/blob/master/img/ibm_real_computer.jpg?raw=true" height=250>
<img src="https://raw.githubusercontent.com/Hacker1337/QML_review/ba8d12d5e8a58542fa0e90fd183bbecde4088006/img/QML_optimization.svg" height=250>
*


This project contains code for running models discussed in a survey article "Survey of Quantum Machine Learning Advances: Insights from the NISQ Era"

It presents several algorithms for quantum computers that have hope of achieving an advantage over classical models, at least in some tasks in the Noisy Intermediate Scale Quantum (NISQ) era or in the nearest future.

The code can be found in the corresponding folders of the project or in publication notebooks attached to the project.


1. **QML frameworks** \
   Tools for running quantum algorithms by simulation or by real quantum computers. Frameworks' abilities and limitations. \
   Our code for benchmarking and comparing different frameworks.
2. **Generative models**
   - **Quantum Recurrent Unit on Gaussian platform** \
     A continuous variable-based algorithm used for text translation in QRNN model.\
     Our implementation of the model.
   - **Quantum GANs** \
     Algorithms with widely believed exponential advantage over classical counterparts. \
     Our implementation of the QuGAN model and authors' implementation of QuMolGAN prepared for running.
3. **Quantum Graph classification models**
   - **All-Subgraphs model** \
   Exponential advantage due to parallel processing of subgraphs in the . \
   Implementation provided by authors.
   - Gaussian boson sampling for natural quantum graph embedding based on the number of perfect matchings in subgraphs.
   - **GraphQNTK**\
   Graph kernel method on quantum computer equivalent to infinite width graph neural network with exponential advantage over classical algorithm.\
   Authors' implementation.
4. **Optuna Hyperparam Optimization** \
   Demonstration of usage of tool for hyperparameter optimization. Allow more fair comparison of quantum models and classical baselines.

\* images from [1](https://www.microsoft.com/en-us/research/uploads/prod/2022/03/Quantum-blog_ChetanNayak_03-2022_1400x788.jpg) and [2](https://pennylane.ai/images/qml/whatisqml/QML_optimization.svg)
