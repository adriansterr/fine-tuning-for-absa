---
base_model:
- Nos-PT/Llama-Carvalho-PT-GL
library_name: transformers
tags:
- mergekit
- merge

---
# pt_gl_slerp_t_05

This is a merge of pre-trained language models created using [mergekit](https://github.com/cg123/mergekit).

## Merge Details
### Merge Method

This model was merged using the [SLERP](https://en.wikipedia.org/wiki/Slerp) merge method.

### Models Merged

The following models were included in the merge:
* [Nos-PT/Llama-Carvalho-PT-GL](https://huggingface.co/Nos-PT/Llama-Carvalho-PT-GL)
* D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2

### Configuration

The following YAML configuration was used to produce this model:

```yaml
models:
  - model: "D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2"
  - model: "Nos-PT/Llama-Carvalho-PT-GL"

base_model: "D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2"

merge_method: slerp
dtype: bfloat16

parameters:
  t: 0.5  # Interpolate halfway between the two models
```
