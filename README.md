

## Contextual Recurrent Neural Network
Classical implementation of quantum model introduced in (Anschuetz et al. 2023) is located in QRNN folder. The model is trained on natural language translation task. With the same memory size quantum model slightly outperforms classical GRU and LSTM recurrent units. Different models with different parameters can be trained in `translate_train.ipynb` notebook. Parameters are configured by `trans_params.yaml` file. 


# Reference
Anschuetz, Eric R., Hong-Ye Hu, Jin-Long Huang, and Xun Gao. 2023. “Interpretable Quantum Advantage in Neural Sequence Learning.” PRX Quantum 4 (2): 020338. https://doi.org/10.1103/PRXQuantum.4.020338.
