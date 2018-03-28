# Sampling Autoencoder
Resolution independent image representation on neural network.

This implementation is based on VAE.
The decoder model represents a function f(u, v, z_1, ..., z_n) which outputs the value at point (u, v) on the image.

## Run
Tested on Python 3.6.3 + Chainer 3.4.0.
You may need CUDA GPU.
```
python3 train.py --gpu 0
```

## Output
### 1st epoch
![epoch 1 original](images/epoch_1_original.png)
Original image.

![epoch 1 reconstructed](images/epoch_1.png)
Reconstructed image rendered as 168x168.

### 50th epoch
![epoch 50 original](images/epoch_50_original.png)
Original image.

![epoch 50 reconstructed](images/epoch_50.png)
Reconstructed image rendered as 168x168 on 50th epoch.

