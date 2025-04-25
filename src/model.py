from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_regression_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Single output for regression
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_absolute_error',
                  metrics=['mean_absolute_error'])
    return model
