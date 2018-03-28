import chainer


class SamplingDecoder(chainer.Chain):
    def __init__(self, input_size):
        super().__init__()
        units = [2 + input_size, 256, 256, 256, 1]
        initializer = chainer.initializers.HeNormal()
        with self.init_scope():
            self.linear1 = chainer.links.Linear(units[0], units[1], initialW=initializer)
            self.linear2 = chainer.links.Linear(units[1], units[2], initialW=initializer)
            self.linear3 = chainer.links.Linear(units[2], units[3], initialW=initializer)
            self.linear4 = chainer.links.Linear(units[3], units[4], initialW=initializer)

    def decode(self, z):
        h = z
        h = self.linear1(h)
        h = chainer.functions.relu(h)
        h = self.linear2(h)
        h = chainer.functions.relu(h)
        h = self.linear3(h)
        h = chainer.functions.relu(h)
        h = self.linear4(h)
        return h

    def __call__(self, z, x0_shape):
        xp = self.xp

        # Generate (u, v) coordinates for every samples.
        u = xp.linspace(-1.0, 1.0, x0_shape[2], dtype=xp.float32)
        v = xp.linspace(-1.0, 1.0, x0_shape[3], dtype=xp.float32)
        uv = xp.broadcast_arrays(u[None, None, :, None], v[None, None, None, :])
        uv = xp.concatenate(uv, 1)
        uv = xp.broadcast_to(uv, (z.shape[0], uv.shape[1], uv.shape[2], uv.shape[3]))

        z = z[:, :, None, None]
        z = chainer.functions.broadcast_to(z, (z.shape[0], z.shape[1], uv.shape[2], uv.shape[3]))

        # Combine them.
        uvz = chainer.functions.concat((uv, z))
        uvz = chainer.functions.transpose(uvz, (0, 2, 3, 1))
        uvz = chainer.functions.reshape(uvz, (-1, uvz.shape[3]))

        y = self.decode(uvz)

        y = chainer.functions.reshape(y, (x0_shape[0], x0_shape[2], x0_shape[3], x0_shape[1]))
        y = chainer.functions.transpose(y, (0, 3, 1, 2))
        return y


class SamplingVAE(chainer.Chain):
    def __init__(self):
        super().__init__()
        units = [28 * 28, 256, 256, 8]
        initializer = chainer.initializers.HeNormal()
        with self.init_scope():
            self.e1 = chainer.links.Linear(units[0], units[1], initialW=initializer)
            self.e2 = chainer.links.Linear(units[1], units[2], initialW=initializer)
            self.e3a = chainer.links.Linear(units[2], units[3], initialW=initializer)
            self.e3b = chainer.links.Linear(units[2], units[3], initialW=initializer)

            self.decoder = SamplingDecoder(units[-1])

    def encode(self, x):
        h = x
        h = self.e1(h)
        h = chainer.functions.relu(h)
        h = self.e2(h)
        h = chainer.functions.relu(h)
        h1 = self.e3a(h)
        h2 = self.e3b(h)
        return h1, h2

    def __call__(self, x0, label):
        xp = self.xp

        x = (x0 > xp.random.uniform(size=x0.shape)).astype(x0.dtype)  # binarize randomly
        mu, ln_var = self.encode(x)

        z = mu
        if chainer.config.train:
            std = chainer.functions.exp(0.5 * ln_var)
            gaussian = xp.random.normal(size=mu.shape)
            z += std * gaussian

        y = self.decoder(z, x0.shape)

        x_loss = chainer.functions.bernoulli_nll(x0, y) / y.size
        z_loss = chainer.functions.gaussian_kl_divergence(mu, ln_var) / mu.size

        chainer.report({
            'x_loss': x_loss,
            'z_loss': z_loss,
        }, self)

        return x_loss / z.shape[1] + z_loss / (x0.shape[1] * x0.shape[2] * x0.shape[3])

    def show(self, x0):
        xp = self.xp

        x = (x0 > xp.random.uniform(size=x0.shape)).astype(x0.dtype)
        z, _ = self.encode(x)

        y = self.decoder(z.data, (x0.shape[0], x0.shape[1], 168, 168))
        y = chainer.functions.sigmoid(y)

        return y
