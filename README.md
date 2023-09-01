

## Contextual Recurrent Neural Network
Classical implementation of quantum model introduced in (Anschuetz et al. 2023) is located in QRNN folder. The model is trained on natural language translation task. With the same memory size quantum model slightly outperforms classical GRU and LSTM recurrent units. Different models with different parameters can be trained in `translate_train.ipynb` notebook. Parameters are configured by `trans_params.yaml` file.

## Quantum GAN

Code to learn QuGAN model from (Stein et al. 2021).

Code can be run as a py file and configured with command line arguments, or as a jupyter notebook.

There are 3 available models, that can be chosen by `model` parameter:\
"c" -- classial GAN \
"q_exp" -- expectation value base quantum model with classical noise \
"q_sample" -- quantum sample based model. Uses quantum randomness

## Reference
Anschuetz, Eric R., Hong-Ye Hu, Jin-Long Huang, and Xun Gao. 2023. “Interpretable Quantum Advantage in Neural Sequence Learning.” PRX Quantum 4 (2): 020338. https://doi.org/10.1103/PRXQuantum.4.020338.

Stein, Samuel A., Betis Baheri, Daniel Chen, Ying Mao, Qiang Guan, Ang Li, Bo Fang, and Shuai Xu. 2021. “QuGAN: A Quantum State Fidelity Based Generative Adversarial Network.” In 2021 IEEE International Conference on Quantum Computing and Engineering (QCE), 71–81. https://doi.org/10.1109/QCE52317.2021.00023.
