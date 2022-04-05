from keras.layers import Dense, Dropout, Flatten, Input, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model, Sequential

import numpy as np
np.random.seed(3934)


class ISF_U_module():
    """docstring for ISF_U_module"""
    batch_size = 128
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5

    def __init__(self, output_dimesion, image_size, image_num):
        super(ISF_U_module, self).__init__()

        # define the DeepContour module
        image_input = Input(shape=(image_size, image_size, 3))
        x = Conv2D(32, 5, strides=(1, 1), activation='relu', padding='same', input_shape=(
            image_size, image_size, 3), name="conv1")(image_input)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Conv2D(48, 5, strides=(1, 1), activation='relu',
                   padding='same', name="conv2")(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Conv2D(64, 5, strides=(1, 1), activation='relu',
                   padding='same', name="conv3")(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Conv2D(128, 5, strides=(1, 1), activation='relu',
                   padding='same', name="conv4")(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        out = Flatten()(x)

        deepContour_model = Model(image_input, out)

        deepContour_model.load_weights(
            'dl_models/deepContour.h5', by_name=True)

        iuput_ = []
        output_ = []
        for n in range(image_num):
            locals()['input_' + str(n)
                     ] = Input(shape=(image_size, image_size, 3))
            iuput_.append(locals()['input_' + str(n)])
            locals()['output_' + str(n)
                     ] = deepContour_model(locals()['input_' + str(n)])
            output_.append(locals()['output_' + str(n)])

        concatenated = concatenate(output_, axis=-1)

        out = Dense(output_dimesion, activation='tanh',
                    name="FC")(concatenated)
        # model.add(Dropout(0.5))
        model = Model(inputs=iuput_, outputs=out)

        for layer in deepContour_model.layers:
            layer.trainable = False
        model.compile(loss='mse', optimizer='rmsprop')
        # plot_model(model, to_file='model.png', show_shapes="true")
        self.model = model

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    def train(self, X_train, V, user_weight, seed):
        for x in range(len(X_train)):
            np.random.seed(seed)
            X_train[x] = np.random.permutation(X_train[x])
        np.random.seed(seed)
        V = np.random.permutation(V)
        np.random.seed(seed)
        user_weight = np.random.permutation(user_weight)
        print("Train...ISF_U module")
        history = self.model.fit(x=X_train, y=V,
                                 verbose=0, batch_size=self.batch_size, epochs=self.nb_epoch,
                                 sample_weight=user_weight)
        return history

    def get_projection_layer(self, image):
        Y = self.model.predict(image, batch_size=len(image[0]))
        return Y
