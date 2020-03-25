# Description

Codes for our article, ["A Pitfall of Learning from User-generated Dataset"](https://arxiv.org/abs/2003.10621), on a type of class noise specific to user-generated datasets (e.g. customer reviews) called **Subjective Class Issue**. We used datasets provided generaously by [Donorschoose.org](https://research.donorschoose.org/t/download-opendata/33), [Yelp Review](https://www.yelp.com/dataset), and [Amazon Fine Food](https://www.kaggle.com/snap/amazon-fine-food-reviews). By following the usage below, you can replicate the results shown in our paper.

# Requirements 
* python3
* jupyter notebook

# Usage

If you would like to run our notebooks, please follow the steps below:
1. Download ["Project Essays"](https://research.donorschoose.org/t/download-opendata/33) provided by Donorschoose.org
1. `git clone` this repo
1. `cd` into the repo
1. open python virtual environment
1. run `pip install -r requirements.txt`
1. in the python virtual environment, open `jupyter notebook`
1. open ipynb files, where the name represents the tasks of each notebook
(*make sure you train doc2vec before you run tsne plots)

# Citation

If you would like to cite our paper or use our code, please cite our paper:
```BibTex
@Article{NemotoShweta20,
  author        = "Kei Nemoto, Shweta Jain",
  title         = "A Pitfall of Learning from User-generated Data: In-depth Analysis of Subjective Class Problem",
  journal       = "arXiv:2003.10621",
  year          = "2020",
}
