# Python Package : Variational Inference
---
## Overview
The purpose of this project is to build an open-source Python package with a JAX backend in which Variational Inference (VI) based models are ready to use for diverse deep learning tasks.
As a 4th year student, my contribution was to build a simple Variational Auto-Encoder (VAE) able to carry out differents tasks over the famous MNIST dataset :
- Data generation
- Anomaly detection

## Code walkthrough
### Initialization and training
This-package-functions loading
```python
from src.data_modeling.VAE import VAE
from src.data_modeling.train import generate_train_step
from src.data_processing import prepare_test_set
from src.data_processing import get__y_true
```

Loading other prerequisited elements
```python
import jax
import flax
import optax
# import orbax

import flax.linen as nn
import jax.numpy as jnp
from jax.typing import ArrayLike

from typing import Tuple, Callable
from math import sqrt

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# ---

import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
```

Constants initialization
```python
batch_size = 16
latent_dim = 32
kl_weight = 0.5
num_classes = 10 # in MNIST dataset
dim_params = 784 # =28x28

seed = 0xffff
key = jax.random.PRNGKey(seed)
```

Data importation
```python
train_dataset = MNIST('../data', train = True, transform=T.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
```

Model initialization
```python
key, model_key = jax.random.split(key)

model = VAE(latent_dim=latent_dim)
optimizer = optax.adamw(learning_rate=1e-4)

train_step, train, params, opt_state = generate_train_step(model_key, model, optimizer, 
								batch_size=batch_size, 
								num_classes=num_classes, 
								dim_params=dim_params)
```

Model training
```python
params, opt_state = train(key, params, freq=500, epochs=10, 
						  opt_state=opt_state, 
						  train_loader=train_loader, 
						  batch_size=batch_size, 
						  train_step=train_step)
```

### Data generation
Generation of 4x4 images of class 0. In other words, the following code will generate 16 images of the digit img_class=0
```python
model.img_gen(key, params, num_classes=num_classes, img_class=0, h=4, w=4)
```

### Anomaly detection
Loading of the MNIST test set in order to assess the model anomaly detection capabilities
```python
test_dataset = MNIST('../data', train = False, transform=T.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
```

Evaluating the average loss over the test set of interest. That score will be used as a threshold accoding to which images with higher loss will be considered as anomalies
```python
total_loss, total_mse, total_kl, loss_distrib = model.evaluate(key, params,
								n_classes=num_classes,
								data_loader=test_loader,
								batch_size=batch_size,
								get_loss_distrib=True)
```

Anomaly detection task
```python
class_of_ref = 0
quantile = 0.99 # by default, should be fine-tuned with cross validation methods

img = prepare_test_set(test_loader.dataset, class_of_ref, mixed_classes=True)

thresh = jnp.quantlie(jnp.asarray(loss_distrib), quantile)
anomalies = model.det_anom(key, params, img, 
			   n_classes=num_classes, 
			   tested_class = class_of_ref,
			   threshold=thresh)
```


Evaluating the model performance on anomaly detection task
```python
y_true = get__y_true(test_loader.dataset, class_of_ref=class_of_ref) # getting the ground truth

print(classification_report(y_true, anomalies["estAnomalie"]))
```
## Potential improvements
- Datasets must be packaged in a DataLoader. There is room to think about a function which would easily let anyone build a DataLoader automatically, or straight about an alternative to DataLoader which has several constraints of use.
- The VAE works much better on the MNIST dataset than on the FashionMNIST dataset. An improvement would consist of widening the array of image dataset to which the VAE could be used with good performance by revising the architecture for instance.
- The VAE architecture is rather simple and is llikely to fail on more complex datasets. As of now, the encoding layer is made of Linear transformations coupled with ReLU transformations only. Even the last decoding layer is a ReLU, while it could be so much other things like sigmoid or tanh.
- The VAE must work with grayscale images. If one has colored images, one must manually convert those images with specific functions like `torchvision.transforms.Grayscale`. An interesting avenue would be to allow the processing of colored images which may ultimately lead to the implementation of a different VAE class if required.
- Images cannot have dimensions different than 28x28 pixels, due to the fixed size of the encoding layer (28x28=784). A solution would be to create a method which allow automatic image resizing (JAX has such a function). A more efficient approach would be to make the size of the encoding layer customizable.
- The VAE works on the raw data better than on the normalized data. There is avenue to investigate on whether its from the VAE architecture, because data is usually normalized before feeding a neural net for better performance. For instance, the addition of a normalization layer would be a good start (see `flax.linen.BatchNorm` or `flax.linen.LayerNorm`).
## Future works
Extend tasks to :
- Image denoising
- Image colorization
Extend VAE application to other data types like :
- Quantitative data
- Qualitative data
- Textual data
- Time series
## References
- Blei, David M. “Build, Compute, Critique, Repeat: Data Analysis with Latent Variable Models.” Annual Review of Statistics and Its Application, vol. 1, no. 1, Jan. 2014, pp. 203–32, https://doi.org/10.1146/annurev-statistics-022513-115657. ‌
- Blei, David M., et al. “Variational Inference: A Review for Statisticians.” Journal of the American Statistical Association, vol. 112, no. 518, Feb. 2017, pp. 859–77, https://doi.org/10.1080/01621459.2017.1285773. Accessed 30 Jan. 2020. ‌
- Kingma, Diederik, and Max Welling. Auto-Encoding Variational Bayes. 2014, arxiv.org/pdf/1312.6114. ‌
