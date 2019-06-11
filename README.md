# image-classify
CLI interface to experiment with image classification models.
Based on https://www.freecodecamp.org/news/how-to-build-the-best-image-classifier-3c72010b3d55/

To run on google colab

- install package from github
```sh
$ !pip install git+https://github.com/fokinpv/image-classify
```

- upload training data and unzip it
```sh
$ unzip -qq training_data.zip
```

- train network
```sh
$ !image-classify train --epochs 25 training_data
```

- try to classify images from a valid dataset
```sh
$ !image-classify classify training_data/valid/painting1/image_001.jpg
```
