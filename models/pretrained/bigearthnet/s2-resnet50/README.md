---
thumbnail: "https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/RSiM_Logo_1.png"
tags:
    - resnet50
    - BigEarthNet v2.0
    - Remote Sensing
    - Classification
    - image-classification
    - Multispectral
library_name: configilm
license: mit
widget:
  - src: example.png
    example_title: Example
    output:
      - label: Agro-forestry areas
        score: 0.000000
      - label: Arable land
        score: 0.000000
      - label: Beaches, dunes, sands
        score: 0.000000
      - label: Broad-leaved forest
        score: 0.000000
      - label: Coastal wetlands
        score: 0.000000
---

[TU Berlin](https://www.tu.berlin/) | [RSiM](https://rsim.berlin/) | [DIMA](https://www.dima.tu-berlin.de/menue/database_systems_and_information_management_group/) | [BigEarth](http://www.bigearth.eu/) | [BIFOLD](https://bifold.berlin/)
:---:|:---:|:---:|:---:|:---:
<a href="https://www.tu.berlin/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/tu-berlin-logo-long-red.svg" style="font-size: 1rem; height: 2em; width: auto" alt="TU Berlin Logo"/>  |  <a href="https://rsim.berlin/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/RSiM_Logo_1.png" style="font-size: 1rem; height: 2em; width: auto" alt="RSiM Logo"> | <a href="https://www.dima.tu-berlin.de/menue/database_systems_and_information_management_group/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/DIMA.png" style="font-size: 1rem; height: 2em; width: auto" alt="DIMA Logo"> | <a href="http://www.bigearth.eu/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/BigEarth.png" style="font-size: 1rem; height: 2em; width: auto" alt="BigEarth Logo"> | <a href="https://bifold.berlin/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/BIFOLD_Logo_farbig.png" style="font-size: 1rem; height: 2em; width: auto; margin-right: 1em" alt="BIFOLD Logo">

# Resnet50 pretrained on BigEarthNet v2.0 using Sentinel-2 bands

<!-- Optional images -->
<!--
[Sentinel-1](https://sentinel.esa.int/web/sentinel/missions/sentinel-1) | [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
:---:|:---:
<a href="https://sentinel.esa.int/web/sentinel/missions/sentinel-1"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/sentinel_2.jpg" style="font-size: 1rem; height: 10em; width: auto; margin-right: 1em" alt="Sentinel-2 Satellite"/> | <a href="https://sentinel.esa.int/web/sentinel/missions/sentinel-2"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/sentinel_1.jpg" style="font-size: 1rem; height: 10em; width: auto; margin-right: 1em" alt="Sentinel-1 Satellite"/>
-->

This model was trained on the BigEarthNet v2.0 (also known as reBEN) dataset using the Sentinel-2 bands. 
It was trained using the following parameters:
- Number of epochs: up to 100 (with early stopping after 5 epochs of no improvement based on validation average 
precision macro)
- Batch size: 512
- Learning rate: 0.001
- Dropout rate: 0.15
- Drop Path rate: 0.15
- Learning rate scheduler: LinearWarmupCosineAnnealing for 1000 warmup steps
- Optimizer: AdamW
- Seed: 42

The weights published in this model card were obtained after 19 training epochs.
For more information, please visit the [official BigEarthNet v2.0 (reBEN) repository](https://git.tu-berlin.de/rsim/reben-training-scripts), where you can find the training scripts.

![[BigEarthNet](http://bigearth.net/)](https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/combined_2000_600_2020_0_wide.jpg)

The model was evaluated on the test set of the BigEarthNet v2.0 dataset with the following results:

| Metric            |       Macro |       Micro |
|:------------------|------------------:|------------------:|
| Average Precision |        0.713706 |        0.858878 |
| F1 Score          |        0.650478 |        0.764773 |
| Precision         | 0.738061 | 0.797413 |

# Example
|             A Sentinel-2 image (true color representation)              |
|:---------------------------------------------------:|
| ![[BigEarthNet](http://bigearth.net/)](example.png) |

| Class labels                                                              |                                                          Predicted scores |
|:--------------------------------------------------------------------------|--------------------------------------------------------------------------:|
| <p> Agro-forestry areas <br> Arable land <br> Beaches, dunes, sands <br> ... <br> Urban fabric </p> | <p> 0.000000 <br> 0.000000 <br> 0.000000 <br> ... <br> 0.000000 </p> |


To use the model, download the codes that define the model architecture from the
[official BigEarthNet v2.0 (reBEN) repository](https://git.tu-berlin.de/rsim/reben-training-scripts) and load the model using the
code below. Note that you have to install [`configilm`](https://pypi.org/project/configilm/) to use the provided code.

```python
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

model = BigEarthNetv2_0_ImageClassifier.from_pretrained("path_to/huggingface_model_folder")
```

e.g.

```python
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

model = BigEarthNetv2_0_ImageClassifier.from_pretrained(
  "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0")
```

If you use this model in your research or the provided code, please cite the following papers:
> K. Clasen, L. Hackel, T. Burgert, G. Sumbul, B. Demir, V. Markl, "[reBEN: Refined BigEarthNet Dataset for Remote Sensing Image Analysis][arxiv]", IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2025.
```bibtex
@inproceedings{clasen2025refinedbigearthnet,
  title={{reBEN}: Refined BigEarthNet Dataset for Remote Sensing Image Analysis},
  author={Clasen, Kai Norman and Hackel, Leonard and Burgert, Tom and Sumbul, Gencer and Demir, Beg{"u}m and Markl, Volker},
  year={2025},
  booktitle={IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
}
```
[arxiv]: https://arxiv.org/abs/2407.03653

> L. Hackel, K. Clasen, B. Demir, "ConfigILM: A General Purpose Configurable Library for Combining Image and Language Models for Visual Question Answering.", SoftwareX 26 (2024): 101731.
```bibtex
@article{hackel2024configilm,
  title={ConfigILM: A general purpose configurable library for combining image and language models for visual question answering},
  author={Hackel, Leonard and Clasen, Kai Norman and Demir, Beg{"u}m},
  journal={SoftwareX},
  volume={26},
  pages={101731},
  year={2024},
  publisher={Elsevier}
}
```
