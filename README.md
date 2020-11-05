# LSHM : LOFAR System Health Management
We combine an autoencoder with k-harmonic clustering to learn features. We combine [Deep K-means](https://arxiv.org/abs/1806.10069) and [K Harmonic means](https://www.hpl.hp.com/techreports/2000/HPL-2000-137.html) to implement deep-K-Harmonic means clustering.

Files included are:

``` lofar_models.py ``` : Methods to read LOFAR H5 data and Autoencoder models.

``` kharmonic_lofar.py ``` : Train K-harmonic autoencoder as well as perform clustering in latent space.
