---
base_model: []
library_name: transformers
tags:
- mergekit
- merge

---
# nearswap_0001

This is a merge of pre-trained language models created using [mergekit](https://github.com/cg123/mergekit).

## Merge Details
### Merge Method

This model was merged using the [NearSwap](https://huggingface.co/alchemonaut/QuartetAnemoi-70B-t0.0001) merge method using D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2 as a base.

### Models Merged

The following models were included in the merge:
* D:/Huggingface/hub/models--VAGOsolutions--Llama-3.1-SauerkrautLM-8b-Instruct/snapshots/221e5f92f9dc5fedaf4556b1be6b20642638643c

### Configuration

The following YAML configuration was used to produce this model:

```yaml
models:
  - model: "D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2"
  - model: "D:/Huggingface/hub/models--VAGOsolutions--Llama-3.1-SauerkrautLM-8b-Instruct/snapshots/221e5f92f9dc5fedaf4556b1be6b20642638643c"

base_model: "D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2"

merge_method: nearswap
dtype: bfloat16

parameters:
  t: 0.001
```
