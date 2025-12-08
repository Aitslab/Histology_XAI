![](../docs/_static/unsup.png)

# Discovering Structure in Latent Representations via Unsupervised Clustering
The latent representations of the tiles, extracted from the final layer of the three-class classifier, were processed using unsupervised dimensionality reduction algorithms such as UMAP and t-SNE. The first two components were then visualized in three ways:

1- Colored by treatment group (Control, ECMO, ECMO+LPS, MV, MV+LPS),

2 -Colored by the model’s predicted classes (low, medium, high), and

3 -Colored by the gold standard labels (low, medium, high) for direct comparison between predicted and true classifications.

4- labeled by original image tiles


The [first](./Histo_unsupervised_featurespace_VGG16.ipynb) notebook documents the complete workflow for VGG16-based models, and the [second](./Histo_unsupervised_featurespace_EffnetB4.ipynb) outlines the corresponding process for EfficientNet-based models.