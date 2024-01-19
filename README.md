# Tensor-Product Transformer for Math Question Answering

## Overview
This repository contains a reimplementation and validation of "Enhancing the Transformer with Explicit Relational Encoding for Math Problem Solving" by Schlag et al. (2019). Our work is centered around the Tensor Product Transformer (TP-Transformer), designed to solve mathematical problems expressed in natural language.

## Reference to Original Work
The original research and code are accessible here:
- [Research Paper](https://arxiv.org/abs/1910.06611)
- [Original GitHub Repository](https://github.com/ischlag/TP-Transformer)

## Repository Structure
This repository is structured to include all necessary components for running and understanding the TP-Transformer model:
- **Jupyter Notebook**: A comprehensive notebook that demonstrates the implementation and usage of the TP-Transformer.
- **Source Python Files**: All the source code files are included for detailed examination and further development.
- **Project Report**: A detailed report of our reimplementation and validation study is available here: [Project Report](https://github.com/pabfr99/Tensor-Product-Transformers-For-Question-Answering/blob/main/QA_Report.pdf) 

## Implementation Overview
- **Dataset**: The implementation uses the Mathematics Dataset, covering a range of mathematical topics and difficulties.
- **Architecture**: Alongside a classic transformer model, we introduce modifications to integrate a role vector in the TP-Transformer's attention mechanism.
- **Training**: Models were trained on both individual and mixed modules, employing the Adam optimizer and adjusting model sizes according to our hardware capabilities.

## Running the Project
To run this project:
1. Clone the repository to your local machine.
2. Install the required dependencies.
3. Explore the Jupyter Notebook for a guided implementation and results overview.
4. Dive into the Python source files for more in-depth understanding and custom modifications.

## Conclusion
This repository serves as a practical implementation and validation of the TP-Transformer model, demonstrating its application in natural language processing for mathematical problem-solving.

## Citation
```
@article{schlag2019enhancing,
  title={Enhancing the Transformer with Explicit Relational Encoding for Math Problem Solving},
  author={Schlag, Imanol and Smolensky, Paul and Fernandez, Roland and Jojic, Nebojsa and Schmidhuber, J{\"u}rgen and Gao, Jianfeng},
  journal={arXiv preprint arXiv:1910.06611},
  year={2019}
}
```
