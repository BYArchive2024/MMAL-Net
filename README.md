# MMAL-Net

This is a PyTorch implementation of the paper ["Multi-branch and Multi-scale Attention Learning for Fine-Grained Visual Categorization (MMAL-Net)"](https://arxiv.org/abs/2003.09150) (Fan Zhang, Meng Li, Guisheng Zhai, Yizhao Liu), and the paper has been accepted by the 27th International Conference on Multimedia Modeling (MMM2021). Welcome to discuss with us in issues!

![avatar](./network.png)

### Table of Contents
- <a href='#requirements'>Requirements</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training L-Net'>Training MMAL-Net</a>
- <a href='#evaluation'>Evaluation</a>
- <a href='#model'>Model</a>
- <a href='#reference'>Reference</a>


## Requirements

You can install the requirements using the following command:
```
poetry install
```

## Datasets
Download the [images.zip](https://drive.google.com/file/d/1CCjt9MznDL1UEicb134eDiKdMTwKyyb9/view?usp=sharing) and copy the contents of the extracted **images** folder into **datasets/FGVC_Aircraft/data/images**

You can also try other fine-grained datasets. 

## Training TBMSL-Net
If you want to train the MMAL-Net, please download the pretrained model of [ResNet-50](https://drive.google.com/open?id=1raU0m3zA52dh5ayQc3kB-7Ddusa0lOT-) and move it to **models/pretrained** before run ``python train.py``. You may need to change the configurations in ``config.py`` if your GPU memory is not enough. The parameter ``N_list`` is ``N1, N2, N3`` in the original paper and you can adjust them according to GPU memory. During training, the log file and checkpoint file will be saved in ``model_path`` directory. 

## Evaluation
If you want to test the MMAL-Net, just run ``python test.py``. You need to specify the ``model_path`` in ``test.py`` to choose the checkpoint model for testing.

## Model
We also provide the checkpoint model trained by ourselves, you can download if from [Google Drive](https://drive.google.com/open?id=13ANynWz7O3QK0RdL4KqASW8X_vMb6V4B) for **CUB-200-2011** or download from [here](https://drive.google.com/file/d/1-LD1Jz6Dh-P6Ibtl17scfrTFQTrW4Zy3/view?usp=sharing) for **FGVC-Aircraft**. If you test on our provided model, you will get 89.6% and 94.7% test accuracy, respectively.

## Reference
If you are interested in our work and want to cite it, please acknowledge the following paper:

```
@misc{zhang2020threebranch,
    title={Multi-branch and Multi-scale Attention Learning for Fine-Grained Visual Categorization},
    author={Fan Zhang and Meng Li and Guisheng Zhai and Yizhao Liu},
    year={2020},
    eprint={2003.09150},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

