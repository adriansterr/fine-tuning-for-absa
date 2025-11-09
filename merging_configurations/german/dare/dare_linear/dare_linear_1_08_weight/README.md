---
base_model:
- unsloth/Meta-Llama-3.1-8B-Instruct
library_name: transformers
tags:
- mergekit
- merge

---
# dare_linear_1_08_weight

This is a merge of pre-trained language models created using [mergekit](https://github.com/cg123/mergekit).

## Merge Details
### Merge Method

This model was merged using the [Linear DARE](https://arxiv.org/abs/2311.03099) merge method using [unsloth/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct) as a base.

### Models Merged

The following models were included in the merge:
* D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2
* D:/Huggingface/hub/models--VAGOsolutions--Llama-3.1-SauerkrautLM-8b-Instruct/snapshots/221e5f92f9dc5fedaf4556b1be6b20642638643c

### Configuration

The following YAML configuration was used to produce this model:

```yaml
models:
  - model: "D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2"
    parameters:
      weight: 1.0
      density: 1.0
  - model: "D:/Huggingface/hub/models--VAGOsolutions--Llama-3.1-SauerkrautLM-8b-Instruct/snapshots/221e5f92f9dc5fedaf4556b1be6b20642638643c"
    parameters:
      weight: 0.8
      density: 1.0

base_model: "unsloth/Meta-Llama-3.1-8B-Instruct"

merge_method: dare_linear
dtype: bfloat16

parameters:
  lambda: 1.0
  rescale: false
```
