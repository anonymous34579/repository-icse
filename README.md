# Delta Debugging for Neural Networks
This repository contains codes for the ICSE'24 submission 

<!--## Prerequisite (Py3 & TF2) 

The code is run successfully using **Python 3.9** and **Tensorflow 2.6**. 

We recommend using **conda** to install the tensorflow-gpu environment:

```shell
$ conda create -n tf2-gpu tensorflow-gpu==2.6.0
$ conda activate tf2-gpu
```

To run code in the jupyter notebook, you should add the kernel manually: 

```shell
$ pip install ipykernel
$ python -m ipykernel install --name tf2-gpu
```

<!-- ## Work Flow
- **Multi-level testing metrics**: fully characterize a DNN model from different angles. 
- **Test case generation algorithms**: magnify the similarities measured by the testing metrics between models.
- **Judging mechanism**: make a ‘yes’/‘no’ judgment on whether the suspect model is a stolen copy.


## Files

- `DeepJudge`: DeepJudge testing framework.
- `train_models`: Train clean models and suspect models. 
- `baselines`: Our implementation of watermarking-based [1,2] and fingerprinting-based [3] techniques. 
- `attacks`: Our implementation of model stealing attacks (fine-tuning, pruning, shuffling [4,5] and extraction [6,7,8]). 

**Reference:** 

```
[1] Uchida et al. "Embedding watermarks into deep neural networks." ICMR 2017. 
[2] Zhang et al. "Protecting intellectual property of deep neural networks with watermarking." AisaCCS 2018.
[3] Cao et al. "IPGuard: Protecting intellectual property of deep neural networks via fingerprinting the classification boundary." AsiaCCS 2021.
[4] Lukas et al. "Sok: How robust is image classification deep neural network watermarking?" S&P 2022.
[5] Yan et al. "And then there were none: Cracking white-box DNN watermarks via invariant neuron transforms." Arxiv 2022. 
[6] Papernot et al. "Practical black-box attacks against machine learning." AsiaCCS 2017.
[7] Orekondy et al. "Knockoff nets: Stealing functionality of black-box models." CVPR 2019.
[8] Yuan et al. "Es attack: Model stealing against deep neural networks without data hurdles." TETCI 2022.
```
--!>
