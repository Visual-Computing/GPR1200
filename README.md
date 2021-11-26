# GPR1200 Dataset
Papper: **GPR1200: A Benchmark for General-Purpose Content-Based Image Retrieval** ([ArXiv]("https://visual-computing.com/"))

**Konstantin Schall, Kai Uwe Barthel, Nico Hezel, Klaus Jung**

[Visual Computing Group HTW Berlin]("https://visual-computing.com/")


<img src="images/gpr_main_pic.jpg" width="100%" title="" alt="teaser"></img>



Similar to most vision related tasks, deep learning models have taken over in the field of content-based image retrieval (CBIR) over the course of the last decade. However, most publications that aim to optimise neural networks for CBIR, train and test their models on domain specific datasets. It is therefore unclear, if those networks can be used as a general-purpose image feature extractor. After analyzing popular image retrieval test sets we decided to manually curate GPR1200, an easy to use and accessible but challenging benchmark dataset with 1200 categories and 10 class examples. Classes and images were manually selected from six publicly available datasets of different image areas, ensuring high class diversity and clean class boundaries.

### Results:
<img src="images/result_table.JPG" width="100%" title="" alt="teaser"></img>

## Download Instructions:

The images are available under this [link](https://visual-computing.com/files/GPR1200/GPR1200.zip). Unziping the content will result in an "images" folder, which contains all 12 000 images. Each filename consists of a combination of the GPR1200 category ID and the original name:
"{category ID}_{original name}.jpg

## Evaluation Protocol:

Images are not devided into query and index sets for evaluation and the full mean average precision value is used as the metric. Instructions and evalution code can be found in this repository.
**Under Construction - Cooming soon**

## License Informations:

This dataset is available for for non-commercial research and educational purposes only and the copyright belongs to the original owners. If any of the images belongs to you and you would like it removed, please kindly inform us, we will remove it from our dataset immediately. Since all images were curated from other publicly available datasets, please visit the respective dataset websites for additional license informations. 

* [Google Landmarks v2](https://github.com/cvdfoundation/google-landmark)
* [iNaturalist](https://www.inaturalist.org/pages/developers) 
* [ImageNet Sketch](https://github.com/HaohanWang/ImageNet-Sketch) 
* [INSTRE](http://123.57.42.89/instre/home.html) 
* [Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/) 
* [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)