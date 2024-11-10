Generic Image Classifier
=======================

Using: Python 3.12.4

You may want to make a conda env for this exact version.
Then, as usual, do this into that environment

```
pip install -r requirements.txt
```

This way, you are not using random versions of Python and the dependencies.

Training On Data
================

These are the current state of the trained neural network:

```
config.json
weights.ckpt
weights.data
```

These weights were created by running

```
python classify.py -train
```

So that a random image such as 4321.jpg can be queried for its classification, like:

```
python classify.py 4321.jpg
```

Giving it training data
=======================

I used Kaggle's `dogs-vs-cats.zip` for the contents of ./train, because
you need lots of images where the label is in the name as a convention.

About 25 thousand images:
```
...
-rw-r--r-- 1 rfielding rfielding  13785 Sep 20  2013 cat.9995.jpg
-rw-r--r-- 1 rfielding rfielding  16855 Sep 20  2013 cat.9996.jpg
-rw-r--r-- 1 rfielding rfielding  18575 Sep 20  2013 cat.9997.jpg
-rw-r--r-- 1 rfielding rfielding   9937 Sep 20  2013 cat.9998.jpg
-rw-r--r-- 1 rfielding rfielding  32621 Sep 20  2013 cat.9999.jpg
-rw-r--r-- 1 rfielding rfielding  10559 Sep 20  2013 cat.999.jpg
-rw-r--r-- 1 rfielding rfielding   6638 Sep 20  2013 cat.99.jpg
-rw-r--r-- 1 rfielding rfielding  16220 Sep 20  2013 cat.9.jpg
-rw-r--r-- 1 rfielding rfielding  32053 Sep 20  2013 dog.0.jpg
-rw-r--r-- 1 rfielding rfielding  16889 Sep 20  2013 dog.10000.jpg
-rw-r--r-- 1 rfielding rfielding   5281 Sep 20  2013 dog.10001.jpg
-rw-r--r-- 1 rfielding rfielding  20778 Sep 20  2013 dog.10002.jpg
-rw-r--r-- 1 rfielding rfielding  39597 Sep 20  2013 dog.10003.jpg
...
```


Drop images into ./train, using naming conventions on the file names.
If ./train contains files that start with "cat." or "dog.", and all such
files have these prefixes, then the neural net will

```
# use this array, in sorted order for labels
["cat","dog"]
```

All images will be resized to a consistent size and color depth.
The prefixes on the file names provide a way to label them.

```
# run 10 epochs of training on ./train directory
python classify.py -train
```

Once trained, you can pick random images to see if the correctly classified the data.

```
# test on some random file that is not in ./train
python classify.py ./test1/11605.jpg
```

Given such a directory of images that it was not trained on, it should
accurately classify the files; given that the real label should be
one that is actually in the training set, ["cat","dog"].


