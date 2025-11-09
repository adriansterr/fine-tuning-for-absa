---
base_model:
- unsloth/Meta-Llama-3.1-8B-Instruct
- ai4bharat/hercule-fr
library_name: transformers
tags:
- mergekit
- merge

---
# dare_linear_1_085

This is a merge of pre-trained language models created using [mergekit](https://github.com/cg123/mergekit).

## Merge Details
### Merge Method

This model was merged using the [Linear DARE](https://arxiv.org/abs/2311.03099) merge method using [unsloth/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct) as a base.

### Models Merged

The following models were included in the merge:
* D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2
* [ai4bharat/hercule-fr](https://huggingface.co/ai4bharat/hercule-fr)

### Configuration

The following YAML configuration was used to produce this model:

```yaml
models:
  - model: "D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2"
    parameters:
      weight: 1.0
      density: 1.0
  - model: "ai4bharat/hercule-fr"
    parameters:
      weight: 0.85
      density: 1.0

base_model: "unsloth/Meta-Llama-3.1-8B-Instruct"

merge_method: dare_linear
dtype: bfloat16

parameters:
  lambda: 1.0
  rescale: false
```
