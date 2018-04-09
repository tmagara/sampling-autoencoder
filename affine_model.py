import chainer


class AffineSamplingDecoder(chainer.Chain):
    def __init__(self, input_size):
        super().__init__()
        units = [2 + input_size, 32, 64, 32, 1]
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

    def __call__(self, z, affine_params, x0_shape, skip_affine=False):
        xp = self.xp

        scale, theta, tx, ty = chainer.functions.split_axis(affine_params, (1, 2, 3), 1)
        scale = chainer.functions.exp(scale)
        cos_theta = chainer.functions.cos(theta)
        sin_theta = chainer.functions.sin(theta)
        sr = chainer.functions.concat((scale * cos_theta, scale * sin_theta, -scale * sin_theta, scale * cos_theta))
        tx = chainer.functions.tanh(tx)
        ty = chainer.functions.tanh(ty)
        t = chainer.functions.concat((tx, ty))

        # Generate (u, v) coordinates for every samples.
        u = xp.linspace(-1.0, 1.0, x0_shape[2], dtype=xp.float32)
        v = xp.linspace(-1.0, 1.0, x0_shape[3], dtype=xp.float32)
        u, v = xp.broadcast_arrays(u[:, None, None], v[None, :, None])
        uv = xp.concatenate((u, v), 2)
        uv = xp.broadcast_to(uv[None], (z.shape[0], uv.shape[0], uv.shape[1], uv.shape[2]))

        #  Affine transform: screen coordinates -> decoder model coordinates
        if not skip_affine:
            t = chainer.functions.broadcast_to(t[:, None, None], uv.shape)
            uv = t + uv

            uv = uv[:, :, :, None]
            sr = chainer.functions.reshape(sr, (sr.shape[0], 1, 1, 2, 2))
            sr = chainer.functions.broadcast_to(sr, (sr.shape[0], uv.shape[1], uv.shape[2], uv.shape[4], 2))
            uv = chainer.functions.matmul(uv, sr)
            uv = chainer.functions.squeeze(uv, (3,))

        uv = chainer.functions.transpose(uv, (0, 3, 1, 2))

        # Combine them.
        z = z[:, :, None, None]
        z = chainer.functions.broadcast_to(z, (z.shape[0], z.shape[1], uv.shape[2], uv.shape[3]))
        uvz = chainer.functions.concat((uv, z))
        uvz = chainer.functions.transpose(uvz, (0, 2, 3, 1))
        uvz = chainer.functions.reshape(uvz, (-1, uvz.shape[3]))

        y = self.decode(uvz)

        y = chainer.functions.reshape(y, (x0_shape[0], x0_shape[2], x0_shape[3], x0_shape[1]))
        y = chainer.functions.transpose(y, (0, 3, 1, 2))

        chainer.report({
            'scale': chainer.functions.sum(scale) / scale.size,
            'theta': chainer.functions.sum(theta) / theta.size,
            'tx': chainer.functions.sum(tx) / tx.size,
            'ty': chainer.functions.sum(ty) / ty.size,
        }, self)
        return y


class AffineSamplingVAE(chainer.Chain):
    def __init__(self):
        super().__init__()
        units = [28 * 28, 392, 196, 98, 49, 8]
        initializer = chainer.initializers.HeNormal()
        zero_initializer = chainer.initializers.Constant(0)
        with self.init_scope():
            self.e1 = chainer.links.Linear(units[0], units[1], initialW=initializer)
            self.e2 = chainer.links.Linear(units[1], units[2], initialW=initializer)
            self.e3 = chainer.links.Linear(units[2], units[3], initialW=initializer)
            self.e4 = chainer.links.Linear(units[3], units[4], initialW=initializer)
            self.e5 = chainer.links.Linear(units[4], units[5] * 2, initialW=initializer)
            self.e5_affine = chainer.links.Linear(units[4], 4, initialW=zero_initializer)

            self.decoder = AffineSamplingDecoder(units[-1])

    def encode(self, x):
        h = x
        h = self.e1(h)
        h = chainer.functions.relu(h)
        h = self.e2(h)
        h = chainer.functions.relu(h)
        h = self.e3(h)
        h = chainer.functions.relu(h)
        h = self.e4(h)
        h = chainer.functions.relu(h)
        h1_2 = self.e5(h)
        h1, h2 = chainer.functions.split_axis(h1_2, (h1_2.shape[1] // 2,), 1)
        h3 = self.e5_affine(h)
        h3 = h3 * chainer.config.user_warm_up
        return h1, h2, h3

    def __call__(self, x0, label):
        xp = self.xp

        x = x0
        if chainer.config.train:
            # x = xp.random.binomial(2, x).astype(x.dtype)    # Not implemented in cupy.
            x = x[None]
            x = (x > xp.random.uniform(size=(2,) + x.shape[1:])).astype(x.dtype)
            x = xp.sum(x, 0) / x.shape[0]

        z_mu, z_ln_var, affine_params = self.encode(x)

        z = z_mu
        if chainer.config.train:
            std = chainer.functions.exp(0.5 * z_ln_var)
            gaussian = xp.random.normal(size=z.shape)
            z += std * gaussian

        y = self.decoder(z, affine_params, x0.shape)

        x_loss = chainer.functions.bernoulli_nll(x0, y) / y.size
        z_loss = chainer.functions.gaussian_kl_divergence(z_mu, z_ln_var) / z.size

        chainer.report({
            'x_loss': x_loss,
            'z_loss': z_loss,
        }, self)

        loss = x_loss / z.shape[1]
        loss += z_loss / (x0.shape[1] * x0.shape[2] * x0.shape[3])
        return loss

    def show(self, x):
        xp = self.xp

        z, z_ln_var, a = self.encode(x)

        y1 = self.decoder(z, a, (x.shape[0], x.shape[1], 84, 84))
        y1 = chainer.functions.sigmoid(y1)

        a_mean = chainer.functions.sum(a, 0, True) / a.shape[0]
        a_mean = chainer.functions.broadcast_to(a_mean, a.shape)
        mask = xp.array([1, 0, 0, 0])
        mask = xp.broadcast_to(mask[None, :], a_mean.shape)
        a_mean *= mask
        y2 = self.decoder(z, a_mean, (x.shape[0], x.shape[1], 84, 84))
        y2 = chainer.functions.sigmoid(y2)

        x = xp.broadcast_to(x[:, :, :, None, :, None], x.shape[0:3] + (3, x.shape[3], 3))
        x = xp.reshape(x, x.shape[0:2] + (x.shape[2] * x.shape[3], x.shape[3] * x.shape[4]))
        return {'a_original': x, 'b_output': y1, 'c_normalized': y2}
