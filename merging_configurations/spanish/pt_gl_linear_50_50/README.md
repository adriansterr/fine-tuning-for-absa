---
base_model:
- Nos-PT/Llama-Carvalho-PT-GL
library_name: transformers
tags:
- mergekit
- merge

---
# pt_gl_linear_50_50

This is a merge of pre-trained language models created using [mergekit](https://github.com/cg123/mergekit).

## Merge Details
### Merge Method

This model was merged using the [Linear](https://arxiv.org/abs/2203.05482) merge method.

### Models Merged

The following models were included in the merge:
* D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2
* [Nos-PT/Llama-Carvalho-PT-GL](https://huggingface.co/Nos-PT/Llama-Carvalho-PT-GL)

### Configuration

The following YAML configuration was used to produce this model:

```yaml
models:
  - model: "D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2"
    parameters:
      weight: 0.50
  - model: "Nos-PT/Llama-Carvalho-PT-GL"
    parameters:
      weight: 0.50

merge_method: linear
dtype: bfloat16
```
