# CPWRANGLE – Towards Efficient and Automated Data Wrangling with Large Language Models

Our paper, which employs Parameter Efficient Tuning (PEFT) methods to automate data wrangling tasks, has been submitted. In this implementation, we adopt the same setup as introduced in the [paper](https://github.com/XiangLi1999/PrefixTuning.git) and code by Li et al. To build PEFT models, we utilize the [PEFT](https://huggingface.co/docs/peft/index) library from Hugging Face as a wrapper.

We have incorporated all experimental setup results in our paper to enhance the understanding of our research. Additionally, we have included the code, enabling the reproduction of our results and facilitating future studies in the same field.

## Existing Results
All results are stored in the `results` folder. The raw log outputs during training, inference, and transferring are organized within their respective folders. We have included utility functions to extract values from these log files. 
A comprehensive summary of all results is compiled in the `results/experimental_result.xlsx` file. Besides, we provide a couple of visualization schemes to help readers better understand the results.

## Reproduciability Instructions
### Training Models
To reproduce the result, you will need to train separate PEFT methods from scratch, which mainly rely on the `model.py`. An demonstrative example can be 
```angular2html
python model.py \
    --data_dir 'data/datasets/entity_matching/structured/iTunes-Amazon' \
    --task entity_matching \
    --epochs 200 \
    --batch_size 16 \
    --lr 1e-3 \
    --peft_type lora \
    # --num_virtual_tokens 50 \
    # --r 8 \
    # --la 8
```
The last three arguments will only have an impact when they are applicable to the specific PEFT methods in use.

We support the following PEFT methods:
- `lora`: learns a low-rank decomposition of the weight update for fine-tuning.
- `prefix tuning`: prepends learnable virtual tokens to both the input and the output of each transformer layer.
- `prompt tuning`: solely focuses on the model input by prepending soft prompts (virtual tokens).
- `p-tuning`: learns a template filled with virtual tokens to which the original input is adapted.
- `finetuning`: trains the entire model for the downstream task.

In practice, our code should be adaptable to all language models (LLMs) with just a few lines requiring adjustment.

### Inference
Once the model is learned, you can use the `inference.py` to generate predictions on the test set, as well as measuring the time efficiency. 



