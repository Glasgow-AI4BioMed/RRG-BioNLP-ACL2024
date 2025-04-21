# Gla-AI4BioMed at RRG24: Visual Instruction-tuned Adaptation for Radiology Report Generation

[![hf_space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/collections/X-iZhang/gla-ai4biomed-at-rrg24-67747a3d615ea14619e7a23e)
[![arXiv](https://img.shields.io/badge/Arxiv-2412.04954-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.04954) 
[![hf_space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green)](https://huggingface.co/datasets/StanfordAIMI/rrg24-shared-task-bionlp)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg?)](https://github.com/X-iZhang/RRG-BioNLP-ACL2024/blob/main/LICENSE) 
[![Visitors](https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fgithub.com%2FX-iZhang%2FRRG-BioNLP-ACL2024&label=Views&countColor=%23f36f43&style=flat)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2FX-iZhang%2FRRG-BioNLP-ACL2024)
<!-- [![Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FX-iZhang%2FRRG-BioNLP-ACL2024a&count_bg=%2300C0FF&title_bg=%23004080&icon=&icon_color=%23FFFFFF&title=Views)](https://hits.seeyoufarm.com) -->

## 🔥 News
- **[20 Jun 2024]** 🏆 Gla-AI4BioMed ranked **4th** place in the Shared Task on Large-Scale Radiology Report Generation @ [BioNLP ACL'24](https://aclanthology.org/2024.bionlp-1.7/)! 🎉
  
## Overview

We introduce a radiology-focused visual language model designed to generate radiology reports from chest X-rays. Building on previous findings that large language models (LLMs) can acquire multimodal capabilities when aligned with pretrained vision encoders, we demonstrate similar potential with chest X-ray images. Our model combines an image encoder with a fine-tuned LLM based on the Vicuna-7B architecture, enabling it to generate different sections of a radiology report with notable accuracy.

![architecture](./assets/architecture.png)


## Contents
- [Install](#install)
- [Model Weights](#model-weights)
- [Quick Start](#quick-start)
    - [Concatenate Images](#concatenate-images)
    - [CLI Inference](#cli-inference)
    - [Script Inference](#script-inference)
- [Data Preparation](#data-preparation)
- [Evaluation](#evaluation)
  
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


## Model Weights
### Libra-v0.5

| Version | Size | Projector | Base LLM | Vision Encoder| Checkpoint |
| ------- | ------- | ------- | ------- | ------- | ------- |
| Libra-0.5 | 7B | MLP-2x | Vicuna-7B | CLIP-L-336px | [libra-v0.5-findings](https://huggingface.co/X-iZhang/libra-v0.5-findings) |
| Libra-0.5 | 7B | MLP-2x | Vicuna-7B | CLIP-L-336px | [libra-v0.5-impressions](https://huggingface.co/X-iZhang/libra-v0.5-impressions) |

*Note: These two models are fine-tuned for `Findings` and `Impression` section generation.*

### Projector weights

These projector weights were pre-trained for visual instruction tuning on chest X-ray to text generation tasks. They can be directly used to initialise your model for multimodal fine-tuning in similar clinical domains.

⚠️ Important Note: For compatibility, please ensure that the *projector type*, *base LLM*, *conv_mode*, and *vision encoder* exactly match those used in our projector pretraining setup. Please also ensure the following settings are correctly configured during instruction tuning:

```Shell
--mm_projector_type mlp2x_gelu \
--mm_vision_select_layer -2 \
--mm_vision_select_feature patch \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
```

| Base LLM | conv_mode | Vision Encoder | Projector | Pretrain Data | Download |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Vicuna-7B| libra_v0 | CLIP-L-336px| MLP-2x | [Findings section](https://huggingface.co/datasets/StanfordAIMI/rrg24-shared-task-bionlp) | [projector](https://huggingface.co/X-iZhang/libra-v0.5-findings/resolve/main/mm_mlp2x_projector_findings.bin?download=true) |
| Vicuna-7B | libra_v0 | CLIP-L-336px | MLP-2x | [Impression section](https://huggingface.co/datasets/StanfordAIMI/rrg24-shared-task-bionlp) | [projector](https://huggingface.co/X-iZhang/libra-v0.5-impressions/resolve/main/mm_mlp2x_projector_impressions.bin?download=true) |

## Quick Start

### Concatenate Images
🧩This model supports multiple images (1 to 4) as input during training. You can use the following method to preprocess and horizontally concatenate multiple images (e.g. generating one report from several diagnostic images):

```Python
from PIL import Image

def concatenate_images(images):
    total_width = sum(img.width for img in images) + 10 * (len(images) - 1)
    height = max(img.height for img in images)

    new_img = Image.new('RGB', (total_width, height), (0, 0, 0))

    current_width = 0
    for img in images:
        new_img.paste(img, (current_width, 0))
        current_width += img.width + 10  # Add a 10px black separator between images

    return new_img

# Load images (make sure the paths are correct or use your own images)
img1 = Image.open('chest_x_ray_example1.jpg')
img2 = Image.open('chest_x_ray_example2.jpg')
img3 = Image.open('chest_x_ray_example3.jpg')
img4 = Image.open('chest_x_ray_example4.jpg')

# Concatenate images
result_img = concatenate_images([img1, img2, img3, img4])

# Save the result
result_img.save('concatenated_chest_x_ray.jpg')
```

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

## Evaluation

To ensure reproducibility and output quality, we evaluate our model using the beam search strategy.

```Shell
python -m libra.eval.eval_vqa_libra \
    --model-path X-iZhang/libra-v0.5-impressions \
    --question-file ./path/to/questions_file.jsonl \
    --image-folder ./path/to/image/folder \
    --answers-file /path/to/answer-file.jsonl \
    --num_beams 10 \
    --length_penalty 2 \
    --num_return_sequences 3 \
    --max_new_tokens 1024 \
    --conv-mode libra_v0
```

You can evaluate models on your custom datasets by converting your dataset to the [JSONL format](https://github.com/X-iZhang/Libra/blob/main/CUSTOM_DATA.md#evaluation-dataset-format) and evaluating using [`eval_vqa_libra.py`](https://github.com/X-iZhang/Libra/blob/main/libra/eval/eval_vqa_libra.py).

Additionally, you can execute the evaluation using the command line. For detailed instructions, see [`libra_eval.sh`](https://github.com/X-iZhang/Libra/blob/main/scripts/eval/libra_eval.sh).

```bash
bash ./scripts/eval/libra_eval.sh beam
```

## Acknowledgments 🙏

We extend our gratitude to the BioNLP 2024 [RRG24 Shared Task](https://stanford-aimi.github.io/RRG24/) organisers for providing the baseline pipeline [ViLMedic](https://vilmedic.app/misc/bionlp24/leaderboard) and curating these challenging and exciting tasks.

Also, we sincerely thank the following projects for their contributions:

* [LLaVA](https://github.com/haotian-liu/LLaVA): A Large Language and Vision Assistant, laying the groundwork for multimodal understanding.
* [FastChat](https://github.com/lm-sys/FastChat): An Open Platform for Training, Serving, and Evaluating Large Language Model based Chatbots.
* [LLaMA](https://github.com/facebookresearch/llama): Open and efficient foundation language models that inspired our core language processing capabilities.

## Citation ✒️

If you find our paper useful in your research and applications, please cite using this BibTeX:
```BibTeX
@inproceedings{zhang-etal-2024-gla,
    title = "Gla-{AI}4{B}io{M}ed at {RRG}24: Visual Instruction-tuned Adaptation for Radiology Report Generation",
    author = "Zhang, Xi  and
      Meng, Zaiqiao  and
      Lever, Jake  and
      Ho, Edmond S.L.",
    editor = "Demner-Fushman, Dina  and
      Ananiadou, Sophia  and
      Miwa, Makoto  and
      Roberts, Kirk  and
      Tsujii, Junichi",
    booktitle = "Proceedings of the 23rd Workshop on Biomedical Natural Language Processing",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.bionlp-1.54/",
    doi = "10.18653/v1/2024.bionlp-1.54",
    pages = "624--634",
}
```
