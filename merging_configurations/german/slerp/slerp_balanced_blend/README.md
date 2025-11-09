---
base_model: []
library_name: transformers
tags:
- mergekit
- merge

---
# slerp_balanced_blend

This is a merge of pre-trained language models created using [mergekit](https://github.com/cg123/mergekit).

## Merge Details
### Merge Method

This model was merged using the [SLERP](https://en.wikipedia.org/wiki/Slerp) merge method.

### Models Merged

The following models were included in the merge:
* D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2
* D:/Huggingface/hub/models--VAGOsolutions--Llama-3.1-SauerkrautLM-8b-Instruct/snapshots/221e5f92f9dc5fedaf4556b1be6b20642638643c

### Configuration

The following YAML configuration was used to produce this model:

```yaml
models:
  - model: "D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2"
  - model: "D:/Huggingface/hub/models--VAGOsolutions--Llama-3.1-SauerkrautLM-8b-Instruct/snapshots/221e5f92f9dc5fedaf4556b1be6b20642638643c"

base_model: "D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2"

merge_method: slerp
dtype: bfloat16

parameters:
  # Balanced blend: favor ABSA model in the middle, German model at start/end
  t: [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.9, 0.85, 0.8,
      0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
```
