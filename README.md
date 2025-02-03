# [Learning to Generate Unit Tests for Automated Debugging]()
[Archiki Prasad](https://archiki.github.io/)\*, [Elias Stengel-Eskin](https://esteng.github.io/)\*, [Justin Chih-Yao Chen](https://dinobby.github.io/), [Zaid Khan](https://zaidkhan.me/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

![Figure of the motivation for UTGen](assets/fig1.png)

## Overview
This repository contains the code for our paper [Learning to Generate Unit Tests for Automated Debugging](). We present UTGen, a data curation and training pipeline for training models for unit test generation, and UTDebug, a debugging pipeline that uses generated unit tests for automated code-debugging using LLMs. In this repo, we provide the code for UTDebug to evaluate unit tests extrinsically and a script to evaluate attack rate, output accuracy, and acc $\cap$ attack on three debugging datasets: HE+Fix, MBPP+Fix, and MBPP+Fix (Hard).

![UTDebug motivation and overview](assets/fig3.png)

## Dependencies
This project is built on Python 3.10.11. All dependencies can be installed via:
```
pip install -r requirements.txt
```

## Scripts and Running UTDebug

## Reference
Please cite our paper as 
```
@article{prasad2025unit,
    title = {Learning to Generate Unit Tests for Automated Debugging},
    author = {Prasad, Archiki and Stengel-Eskin, Elias and Chen, Justin Chih-Yao and Khan, Zaid and Bansal, Mohit}, 
    year = {2025},
    journal={arXiv preprint 2502.} 
}
```
