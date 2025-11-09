---
base_model: []
library_name: transformers
tags:
- mergekit
- merge

---
# linear_layer_specific

This is a merge of pre-trained language models created using [mergekit](https://github.com/cg123/mergekit).

## Merge Details
### Merge Method

This model was merged using the [Linear](https://arxiv.org/abs/2203.05482) merge method.

### Models Merged

The following models were included in the merge:
* D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2
* D:/Huggingface/hub/models--VAGOsolutions--Llama-3.1-SauerkrautLM-8b-Instruct/snapshots/221e5f92f9dc5fedaf4556b1be6b20642638643c

### Configuration

The following YAML configuration was used to produce this model:

```yaml
merge_method: linear
parameters:
  weight: 0.8  # Default weight for finetuned model

# Override specific layers to get more German knowledge
slices:
  - sources:
      - model: "D:/Uni/Masterarbeit Code/jakob_finetuning/finetuned_models/meta_llama_full_colab_remerge_2"
        layer_range: [0, 16]  # Early layers: more finetuned
      - model: "D:/Huggingface/hub/models--VAGOsolutions--Llama-3.1-SauerkrautLM-8b-Instruct/snapshots/221e5f92f9dc5fedaf4556b1be6b20642638643c"
        layer_range: [16, 32]  # Later layers: more German
    parameters:
      weight: [0.9, 0.1]  # Still favor finetuned
```
