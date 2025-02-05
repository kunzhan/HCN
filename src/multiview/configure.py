def get_default_config(data_name):
    #48, 40, 254, 1984, 512, 928
    if data_name in ['Caltech101-20']:
        return dict(
            Autoencoder = dict(
                arch1 = [254, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                arch2 = [1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch3 = [512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1 = 'relu',
                activations2 = 'relu',
                activations3 = 'relu',
                batchnorm = True,
            ),
            training = dict(
                seed = 70,
                batch_size = 128,
                epoch = 100,
                lr = 1.0e-4,
                # Balanced factors for L_cls, L_code, and L_glb
                alpha = 3,
                beta= 3.2,
                gamma= 7.8,
                lambda1 = 0.02,
                lambda2 = 0.01,
                droprate = 0.10,
            ),
        )
    elif data_name in ['Scene_15']:
        """The default configs."""
        return dict(
            Autoencoder = dict(
                arch1 = [20, 1024, 1024, 1024, 128],
                arch2 = [59, 1024, 1024, 1024, 128],
                arch3 = [40, 1024, 1024, 1024, 128],
                activations1 = 'relu',
                activations2 = 'relu',
                activations3 = 'relu',
                batchnorm = True,
            ),
            training = dict(
                seed = 1,
                batch_size = 256,
                epoch = 200,
                lr = 1.0e-4,
                # Balanced factors for L_cls, L_code, and L_glb
                alpha = 3.2,
                beta= 2.0,
                gamma= 3.0,
                lambda1 = 0.01,
                lambda2 = 1,
                droprate = 0.08,
            ),
        )


    elif data_name in ['LandUse_21']:
        """The default configs."""
        return dict(
            Autoencoder = dict(
                arch1 = [20, 1024, 1024, 1024, 64],
                arch2 = [59, 1024, 1024, 1024, 64],
                arch3 = [40, 1024, 1024, 1024, 64],
                activations1 = 'relu',
                activations2 = 'relu',
                activations3 = 'relu',
                batchnorm = True,
            ),
            training = dict(
                seed = 20,
                epoch = 300,
                batch_size = 128,
                lr = 1.0e-4,
                # Balanced factors for L_cls, L_code, and L_glb
                alpha = 3,
                beta=4.4,
                gamma=8.0,
                lambda1 = 0.01,
                lambda2 = 5,
                droprate = 0.08,
            ),
        )
    else:
        raise Exception('Undefined data_name')
