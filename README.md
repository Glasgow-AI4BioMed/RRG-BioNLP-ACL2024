# Gla-AI4BioMed at RRG24: Visual Instruction-tuned Adaptation for Radiology Report Generation
[![arXiv](https://img.shields.io/badge/Arxiv-2412.04954-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.04954) 
[![hf_space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green)](https://huggingface.co/datasets/StanfordAIMI/rrg24-shared-task-bionlp)
[![hf_space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/collections/X-iZhang/gla-ai4biomed-at-rrg24-67747a3d615ea14619e7a23e)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg?)](https://github.com/X-iZhang/RRG-BioNLP-ACL2024/blob/main/LICENSE) 

## Overview

We introduce a radiology-focused visual language model designed to generate radiology reports from chest X-rays. Building on previous findings that large language models (LLMs) can acquire multimodal capabilities when aligned with pretrained vision encoders, we demonstrate similar potential with chest X-ray images. Our model combines an image encoder with a fine-tuned LLM based on the Vicuna-7B architecture, enabling it to generate different sections of a radiology report with notable accuracy.
![architecture](./assets/architecture.png)

## Project

Please refer to the [**Libra repository**](https://github.com/X-iZhang/Libra) for code and environment details, as this project is compatible with it. Below is a brief outline:

1. **Environment Setup**  
   - Create and activate a new conda environment (e.g., `libra`).
   - Install the required dependencies (e.g., `pip install -e .`).  
   - For more detailed instructions, see [Libra's README](https://github.com/X-iZhang/Libra/blob/main/README.md).

2. **Model weight**
   
| Version | Base LLM | Vision Encoder| Checkpoint |
| ------- | ------- | ------- | ------- |
| Libra-v0.5-impressions| Vicuna-7B | CLIP | [libra-v0.5-impressions](https://huggingface.co/X-iZhang/libra-v0.5-impressions) |
| Libra-v0.5-findings | Vicuna-7B | CLIP | [libra-v0.5-findings](https://huggingface.co/X-iZhang/libra-v0.5-findings) |

3. **Data Preparation**  
   - Refer to the instructions in **Libra** regarding data structure, preprocessing, and any additional script usage.


## Acknowledgments

We extend our gratitude to the BioNLP 2024 RRG24 Shared Task organisers for providing the baseline pipeline ViLMedic and curating these challenging and exciting tasks.

## Citation

If you find our paper useful in your research and applications, please cite using this BibTeX:
```BibTeX
@inproceedings{Zhang_2024,
   title={Gla-AI4BioMed at RRG24: Visual Instruction-tuned Adaptation for Radiology Report Generation},
   url={http://dx.doi.org/10.18653/v1/2024.bionlp-1.54},
   DOI={10.18653/v1/2024.bionlp-1.54},
   booktitle={Proceedings of the 23rd Workshop on Biomedical Natural Language Processing},
   publisher={Association for Computational Linguistics},
   author={Zhang, Xi and Meng, Zaiqiao and Lever, Jake and Ho, Edmond S.L.},
   year={2024},
   pages={624–634}
}
```
