# Fusion IDLab
Official implementation of "[Harmonizing Image Forgery Detection &amp; Localization: Fusion of Complementary Approaches](https://media.idlab.ugent.be/fusion-forgery-detection)"

## Try-out without downloading or cloning
The fusion method is integrated in the [COM-PRESS dashboard](https://com-press.ilabt.imec.be/home), where anyone can upload images for manipulation analysis.

![Example of FusionIDLab on COM-PRESS dashboard](https://media.idlab.ugent.be/images/posts/2023-12-25-fusion-forgery-detection.jpg)


## Examples
Below, some visual examples of the heatmaps of the individual methods and the proposed fusion method are given.
![Examples of individual heatmaps and output of Fusion](https://media.idlab.ugent.be/images/posts/2023-12-25-fusion-forgery-detection-01.jpg)

Additionally, the examples_input and examples_output folders contain an example that is used in the Jupyter notebook _run_forgery_detection_fusion.ipynb_.

## Dependencies
The code requires Python 3.X and was built with Tensorflow 2.15. Additionally, there are dependencies to two git submodules, comprint and CAT-Net, which require tensorflow and torch, respectively.

Install the requested libraries using:
```
pip install -r requirements_tf.txt
pip install -r requirements_torch_catnet.txt
```
## Model weights
The fusion model weights can be downloaded by running the _download_fusion_weights.sh_ script from the root fusion-idlab folder. The CAT-Net weights can be downloaded by running the _download_weights.sh_ script from the CAT-Net folder. The comprint weights are included in the corresponding repository.

## Usage (inference)
The Jupyter notebook _run_forgery_detection_fusion.ipynb_ gives an example on how to run all individual methods, as well as the proposed fusion method. By adding new files in _examples_input_ and changing the path in the notebook, you can extract the heatmaps from other images under investigation.

## More information
More information can be found on [our website](https://media.idlab.ugent.be/fusion-forgery-detection).

The paper can be downloaded [here](https://doi.org/10.3390/jimaging10010004).

## Reference
This work was published in [Journal of Imaging](https://www.mdpi.com/journal/jimaging).

```js
@Article{mareen2024fusion,
  AUTHOR = {Mareen, Hannes and De Neve, Louis and Lambert, Peter and Van Wallendael, Glenn},
  TITLE = {Harmonizing Image Forgery Detection & Localization: Fusion of Complementary Approaches},
  JOURNAL = {Journal of Imaging},
  VOLUME = {10},
  YEAR = {2024},
  NUMBER = {1},
  ARTICLE-NUMBER = {4},
  URL = {https://www.mdpi.com/2313-433X/10/1/4},
  PubMedID = {38248989},
  ISSN = {2313-433X},
  DOI = {10.3390/jimaging10010004}
}
```
