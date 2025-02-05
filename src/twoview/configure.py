def get_default_config(data_name):
    if data_name in ['Caltech101-20']:
        return dict(
            Autoencoder = dict(
                arch1 = [1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2 = [512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1 = 'relu',
                activations2 = 'relu',
                batchnorm = True,
            ),
            training = dict(
                seed = 4,
                batch_size = 128,
                epoch = 100,
                lr = 1.0e-4,
                # Balanced factors for L_cls, L_code, and L_glb
                alpha = 3,
                beta= 3,
                gamma= 8,
                lambda1 = 0.1,
                lambda2 = 0.1,
                droprate = 0.10,
            ),
        )
    elif data_name in ['Scene_15']:
        """The default configs."""
        return dict(
            Autoencoder = dict(
                arch1 = [20, 1024, 1024, 1024, 128],
                arch2 = [59, 1024, 1024, 1024, 128],
                activations1 = 'relu',
                activations2 = 'relu',
                batchnorm = True,
            ),
            training = dict(
                seed = 10,
                batch_size = 256,
                epoch = 200,
                lr = 1.0e-4,
                # Balanced factors for L_cls, L_code, and L_glb
                alpha = 3.8,
                beta= 2.7,
                gamma= 2.2,
                lambda1 = 0.01,
                lambda2 = 1,
                droprate = 0.08,
            ),
        )

    elif data_name in ['NoisyMNIST']:
        """The default configs."""
        return dict(
            Autoencoder = dict(
                arch1 = [784, 1024, 1024, 1024, 64],
                arch2 = [784, 1024, 1024, 1024, 64],
                activations1 = 'relu',
                activations2 = 'relu',
                batchnorm = True,
            ),
            training = dict(
                seed = 2,
                epoch = 100,
                batch_size = 256,
                lr = 1.0e-4,
                # Balanced factors for L_cls, L_code, and L_glb
                alpha = 3,
                beta= 3,
                gamma= 8,
                lambda1 = 0.3,
                lambda2 = 0.01,
                droprate = 0.10,
            ),
        )

    elif data_name in ['LandUse_21']:
        """The default configs."""
        return dict(
            Autoencoder = dict(
                arch1 = [59, 1024, 1024, 1024, 64],
                arch2 = [40, 1024, 1024, 1024, 64],
                activations1 = 'relu',
                activations2 = 'relu',
                batchnorm = True,
            ),
            training = dict(
                seed = 20,
                epoch = 300,
                batch_size = 128,
                lr = 1.0e-4,
                # Balanced factors for L_cls, L_code, and L_glb
                alpha = 3,
                beta=3.6,
                gamma=9.5,
                lambda1 = 0.01,
                lambda2 = 5,
                droprate = 0.08,
            ),
        )
    else:
        raise Exception('Undefined data_name')
