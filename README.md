# Gla-AI4BioMed at RRG24: Visual Instruction-tuned Adaptation for Radiology Report Generation

[![hf_space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/collections/X-iZhang/gla-ai4biomed-at-rrg24-67747a3d615ea14619e7a23e)
[![arXiv](https://img.shields.io/badge/Arxiv-2412.04954-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.04954) 
[![hf_space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green)](https://huggingface.co/datasets/StanfordAIMI/rrg24-shared-task-bionlp)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg?)](https://github.com/X-iZhang/RRG-BioNLP-ACL2024/blob/main/LICENSE) 
[![Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FX-iZhang%2FRRG-BioNLP-ACL2024a&count_bg=%2300C0FF&title_bg=%23004080&icon=&icon_color=%23FFFFFF&title=Views)](https://hits.seeyoufarm.com)

## Overview

We introduce a radiology-focused visual language model designed to generate radiology reports from chest X-rays. Building on previous findings that large language models (LLMs) can acquire multimodal capabilities when aligned with pretrained vision encoders, we demonstrate similar potential with chest X-ray images. Our model combines an image encoder with a fine-tuned LLM based on the Vicuna-7B architecture, enabling it to generate different sections of a radiology report with notable accuracy.

![architecture](./assets/architecture.png)


## Install

Please refer to the [**Libra repository**](https://github.com/X-iZhang/Libra) for code and environment details, as this project is compatible with it. Below is a brief outline:

- Create and activate a new conda environment (e.g., `libra`).
- Install the required dependencies (e.g., `pip install -e .`).  

```Shell
git clone https://github.com/X-iZhang/Libra.git
cd Libra

conda create -n libra python=3.10 -y
conda activate libra
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

- For more detailed instructions, see [Libra's README](https://github.com/X-iZhang/Libra/tree/main#install).


## **Model weight**
   
| Version | Base LLM | Vision Encoder| Checkpoint |
| ------- | ------- | ------- | ------- |
| Libra-v0.5-impressions| Vicuna-7B | CLIP | [libra-v0.5-impressions](https://huggingface.co/X-iZhang/libra-v0.5-impressions) |
| Libra-v0.5-findings | Vicuna-7B | CLIP | [libra-v0.5-findings](https://huggingface.co/X-iZhang/libra-v0.5-findings) |

## Quick Start

### CLI Inference
We support running inference using the CLI. To use our model, run:
```Shell
python -m libra.serve.cli \
    --model-path X-iZhang/libra-v0.5-impressions  \
    --conv-mode libra_v0 \
    --image-file "./path/to/chest_x_ray.jpg"
```

### Script Inference
You can use the `libra_eval` function in `libra/eval/run_libra.py` to easily launch a model trained by yourself or us on local machine or in Google Colab, after installing this repository.

```Python
from libra.eval import libra_eval

model_path = "X-iZhang/libra-v0.5-impressions "  # Or "X-iZhang/libra-v0.5-findings " 

# Define the paths to the images. 
image_files = "./path/to/chest_x_ray.jpg"

# Define the prompt to guide the model's response.
prompt = "Provide a detailed description of the impression in the radiology image. " 
# Or  "Provide a detailed description of the findings in the radiology image. " 

# Specify the conversational mode, matching the PROMPT_VERSION used during training.
conv_mode = "libra_v0"

# Call the libra_eval function.
libra_eval(
    model_path=model_path,
    image_file=image_files,
    query=prompt,
    temperature=0.9,
    top_p=0.8,
    conv_mode=conv_mode,
    max_new_tokens=512
)
```

## Data Preparation
We use the officially provided [dataset](https://huggingface.co/datasets/StanfordAIMI/rrg24-shared-task-bionlp). For information on data structure, preprocessing, and additional script usage, please refer to the instructions in **Libra**. For detailed formats related to data training or evaluation, see [`Custom_Data.md`](https://github.com/X-iZhang/Libra/blob/main/CUSTOM_DATA.md).

## Acknowledgments 🙏

We extend our gratitude to the BioNLP 2024 [RRG24 Shared Task](https://stanford-aimi.github.io/RRG24/) organisers for providing the baseline pipeline [ViLMedic](https://vilmedic.app/misc/bionlp24/leaderboard) and curating these challenging and exciting tasks.

Also, we sincerely thank the following projects for their contributions:

* [LLaVA](https://github.com/haotian-liu/LLaVA): A Large Language and Vision Assistant, laying the groundwork for multimodal understanding.
* [FastChat](https://github.com/lm-sys/FastChat): An Open Platform for Training, Serving, and Evaluating Large Language Model based Chatbots.
* [LLaMA](https://github.com/facebookresearch/llama): Open and efficient foundation language models that inspired our core language processing capabilities.

## Citation ✒️

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
