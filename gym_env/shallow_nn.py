from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


class ShallowNN(Model):
    def __init__(self, fc1_dims=128, out_dims=1):
        super(ShallowNN, self).__init__()
        self.fc1_dims = fc1_dims
        self.out_dims = out_dims

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.output_layer = Dense(self.out_dims)

    def call(self, x):
        x = self.fc1(x)
        return self.output_layer(x)
