# Feature Extractor tests on the UCMerced dataset

Below are visualizations of the different feature extractors in `patchwork` trained on the [UC Merced Land Use dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html). I used the same network design (VGG19) for each, and computed average feature vectors for the full dataset. Then I plotted  T-SNE embeddings and computed test accuracies with a multiclass logistic regression.

Note that UCMerced has only 2100 examples, and self-supervision techniques [tend to improve with more data](http://openaccess.thecvf.com/content_CVPR_2019/html/Kolesnikov_Revisiting_Self-Supervised_Visual_Representation_Learning_CVPR_2019_paper.html), so these differences might not extend to larger datasets.

## Baseline: VGG19 pretrained on ImageNet

Test accuracy: 87.6%

![](feature/imagenet_tsne.png)


## VGG19 trained with Context Encoder

Test accuracy: 81.0%

![](feature/context_encoder_tsne.png)


## VGG19 trained with DeepCluster

Test accuracy: 93.7%

![](feature/deepcluster_tsne.png)


## VGG19 trained with SimCLR

Test accuracy: 90.1%

![](feature/simclr_tsne.png)