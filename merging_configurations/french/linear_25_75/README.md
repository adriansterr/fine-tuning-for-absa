---
base_model:
- ai4bharat/hercule-fr
library_name: transformers
tags:
- mergekit
- merge

---
# linear_25_75

This is a merge of pre-trained language models created using [mergekit](https://github.com/cg123/mergekit).

## Merge Details
### Merge Method

This model was merged using the [Linear](https://arxiv.org/abs/2203.05482) merge method.

### Models Merged

The following models were included in the merge:
* [ai4bharat/hercule-fr](https://huggingface.co/ai4bharat/hercule-fr)
* D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2

### Configuration

The following YAML configuration was used to produce this model:

```yaml
models:
  - model: "D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2"
    parameters:
      weight: 0.25  
  - model: "ai4bharat/hercule-fr"
    parameters:
      weight: 0.75  

merge_method: linear
dtype: bfloat16
```
