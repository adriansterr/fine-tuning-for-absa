---
base_model:
- ai4bharat/hercule-fr
library_name: transformers
tags:
- mergekit
- merge

---
# slerp_t_05

This is a merge of pre-trained language models created using [mergekit](https://github.com/cg123/mergekit).

## Merge Details
### Merge Method

This model was merged using the [SLERP](https://en.wikipedia.org/wiki/Slerp) merge method.

### Models Merged

The following models were included in the merge:
* [ai4bharat/hercule-fr](https://huggingface.co/ai4bharat/hercule-fr)
* D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2

### Configuration

The following YAML configuration was used to produce this model:

```yaml
models:
  - model: "D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2"
  - model: "ai4bharat/hercule-fr"

base_model: "D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2"

merge_method: slerp
dtype: bfloat16

parameters:
  t: 0.5  # Interpolate halfway between the two models
```
