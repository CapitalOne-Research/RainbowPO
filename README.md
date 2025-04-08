# RainbowPO : A Unified Framework for Combining Improvements in Preference Optimization

This is the code repository of our [ICLR 2025 paper](https://openreview.net/forum?id=trKee5pIFv).
### Our base codebase is build on a modified version of [Huggingface TRL](https://github.com/huggingface/trl)

### We also have embedded `alpaca_eval` from https://github.com/tatsu-lab/alpaca_eval inside our repo for reproducibility.

## Preparation

For both training and evaluations, the images on 0710 and 0731 should both be OK, training is previously done on `<some image>`, and evaluation has been tested on `<some image>`.

### 1. Download the base model to be finetuned
Download a model (e.g., Llama3-8B-Instruct), which you are going to fine-tune, and set the path to `model_path`.

### 2. Data preparation
Download datasets from [Ultrafeedback Armorm](https://huggingface.co/datasets/princeton-nlp/llama3-ultrafeedback-armorm?row=0) created by SimPO authors,
and transform it into the trl format (which is needed to successfully run the code for trl scripts). The transformation script is .

## Training

### 1. Change the hyperparameters in the DPOconfig
For current implementation (by 08/22/2024), we first need to manually change the setups in the DPOconfig. Find the `dpo_config.py` under `trl\trainer\`. The
main hyperparameters needed to change is 
1. $\beta$: float type, which represents the penalty constant, default is 0.1 for dpo
2. home advantage: float type, which is the $\gamma$ factor, default is 0 for dpo
3. if_mixing_alpha: bool, which represents whether mixing reference policy and a fix home advantage as in the RainbowPO design, default is False for dpo
4. mixing_alpha: float type, which is a constant within 0 and 1, default is 0.5
5. length_normalization: bool type, whether to apply length normalization in dpo
6. reference free: bool type, whether to ignore the reference policy term in the objective.
7. neg_log_dispersion_mean: the scaling factor if applying Mallows-DPO.
8. add_sft_loss: bool type, whether to add sft loss.
9. sft_coef: float type, the sft loss constant
10. loss_type: choose on in the loss list

### An automatic script to change the hyperparameters in this file

### 2. Run the scripts under `examples\`

1. Launch a pod.
2. cd to `\RainbowPO\`
3. `pip install -e .`
4. `pip install deepspeed==0.14.5` (notice that the most recent deepspeed is 0.15.0, which conflicts with trl)
5. `pip install wandb==0.17.5`, `wandb disabled`
6. check in different .ipynbs for the accelerate commands to run, including the configs.

## Evaluations

1. cd alpaca_eval
2. `pip install -e .`
3. `pip install vllm==0.5.3`
4. if met error like ```libstdc++.so.6: version `GLIBCXX_3.4.29' not found```, this can be solved `pip install --upgrade --force-reinstall zmq` as in [here](https://github.com/pybind/pybind11/discussions/3453).
5. modify the RainbowPO model configs under model_config, which gives the `model_name` for later evaluation by Llama3-70B-Instruct.
6. `CUDA_VISIBLE_DEVICES=0 alpaca_eval generate_from_model --model_configs "Llama-3-Instruct-8B-RainbowPO"`
7. do 1-3 again in another terminal the latest opened cpu pod/notebook and run `alpaca_eval evaluate --reference_outputs='<some fsx path>/evaluations/alpaca_eval/results/gpt4_1106_preview/model_outputs.json' --model_outputs='<some fsx path>>/evaluations/alpaca_eval/results/{model_name}/model_outputs.json'  --annotators_config='alpaca_eval_llama3_70b_fn_local'` in which the reference outputs is taken as gpt4_1106_preview generated answers for default and model outputs is from the model name to be saved. The win rate will then be calculated.

## Citation

Please use the following bibtex to cite out paper: 
```
@inproceedings{zhao2025rainbowpo,
    title={Rainbow{PO}: A Unified Framework for Combining Improvements in Preference Optimization},
    author={Hanyang Zhao and Genta Indra Winata and Anirban Das and Shi-Xiong Zhang and David Yao and Wenpin Tang and Sambit Sahu},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=trKee5pIFv}
}
```
