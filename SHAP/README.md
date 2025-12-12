![](../docs/_static/shap.png)

# SHAP (SHapley Additive exPlanations) 

In this [section](Histo_Shap_values.ipynb), the code 

Feeds each tile into a pretrained model

Generates SHAP explanations (activation/importance maps)

Visualizes both the tile and its SHAP explanation

To perform XAI (explainable AI) on image tiles.


we have created  SHAP image masker
``` python
masker = shap.maskers.Image("inpaint_ns", tmp_img.shape)
```
SHAP masked parts of the image. "inpaint_ns" means it inpaints masked regions with a noise-based method. This helps evaluate which regions(pixels) contribute to the output.

We then Created the SHAP explainer:

``` python
explainer = shap.Explainer(pretrained_model, masker, output_names=['low','medium','high'])
```
where 
pretrained_model is the ML model We are explaining (En_m_a)

masker tells SHAP how to perturb the image.

output_names labels the model’s output classes.

The I have Computed SHAP values as:
```python
shap_values = explainer(y, outputs=shap.Explanation.argsort.flip[:3])
```
and  Produced SHAP attribution maps for the model’s top 3 predicted classes where this tells us which pixels influenced the model.

✅ Summary

This section of code does the following for each 224×224 tile:

Extract tile

Visualize tile

Prepare it for model input

Compute SHAP explanations

Visualize SHAP heatmaps

Print the model's prediction

It is essentially an explainability pipeline for tile-based image classification.