# SAViR-T: Spatially Attentive Visual Reasoning with Transformers
This repository contains the codebase of "SAViR-T: Spatially Attentive Visual Reasoning with Transformers," which was published at ECML/PKDD 2022.

## SAViR-T model

<center>
<img src="figures/savirt.png" width="75%">
</center>

SAViR-T consists of a **Backbone Network**, a **Visual Transformer**, and a **Reasoning Network**. Each puzzle image goes through the Backbone Network ($\Phi_{\text{CNN}}$) to extract the **Region Feature Maps**. The Visual Transformer processes the features of local image patches and generates the attended vectors for each image. Then, the Reasoning Network operates on a per patch/token basis (represented by $K$ parallel layers/tokens) for the context and choice list attended vectors. For each token, we first create complete rows by filling the third row of the context matrix with images from the choice list ($a=9,\dots,16$). Next, we extract row features using the **Row Rule Extraction** module ($\Phi_{\text{MLP}}$) followed by row-pair representations using the **Shared Rule Extraction** MLP network ($\Psi_{\text{MLP}}$). Then, the **Principal Shared Rule** ($r^{12}$) is compared against each remaining Shared Rule derived from the puzzle-filled third row using choice list images. Finally, considering all tokens (local patches), the choice list image with the highest similarity to the Principal Shared Rule is selected as the puzzle answer.

## Installation and Setup
To clone the repository, use:
```bash
git clone https://github.com/kalbasioti/savir-t.git
```
In our experiments we used Python 3.12.4. For replicating our python environment see `requirements.txt` file.


## Preparing V-PROM Dataset

The next step, for using our codebase, is to download the [V-PROM](https://ojs.aaai.org/index.php/AAAI/article/view/6885) dataset. For the V-PROM images, extract the features before the last average pooling layer using the pretrained [ResNet-101](https://pytorch.org/vision/0.12/generated/torchvision.models.resnet101.html). Save the extracted features in a train and a test folder.

Copy the V-PROM files `attr_gt_labels.npy`, `counting_gt_labels.npy` and `obj_gt_labels.npy` in the `datasets` folder.


## Training
For training SAViR-T, use:
```
python main.py --train_data <train_data_folder_location> --test_data <test_data_folder_location>
```
For the `<train_data_folder_location>` and `<test_data_folder_location>` replace them with your local paths to the V-PROM dataset extracted image features.


## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [SRAN](https://github.com/husheng12345/SRAN)
- [ViT](https://github.com/lucidrains/vit-pytorch/tree/main)


## Citing
If you find this code useful for your research, please consider citing the following paper:

```bibtex
@inproceedings{sahu2022savir,
  title={Savir-t: Spatially attentive visual reasoning with transformers},
  author={Sahu, Pritish and Basioti, Kalliopi and Pavlovic, Vladimir},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={460--476},
  year={2022},
  organization={Springer}
}
```