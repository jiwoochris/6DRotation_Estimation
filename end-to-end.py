import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load your labels
df = pd.read_csv('new_xy_data.csv')

# Create a generator for loading images
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

def generator(df, directory, x_col, y_cols, batch_size):
    gen = datagen.flow_from_dataframe(
        df,
        directory=directory,
        x_col=x_col,
        y_col=y_cols,
        class_mode='raw',
        target_size=(224, 224),
        batch_size=batch_size
    )
    return gen

# Use this generator to create your training and validation sets
# For example, if you want to use 90% of your data for training
train_df = df.sample(frac=0.9, random_state=0)
val_df = df.drop(train_df.index)

# Use the generator function to create your data loaders
train_gen = generator(train_df, 'e_data', 'ImageName', ['x', 'y'], 32)
val_gen = generator(val_df, 'e_data', 'ImageName', ['x', 'y'], 32)





from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Define the model architecture
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='linear')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error')

# Train the model
model.fit(train_gen, validation_data=val_gen, epochs=20)




# Evaluate the model on the validation set
val_loss = model.evaluate(val_gen)
print(f"Validation loss: {val_loss}")




# Predict the coordinates of a new image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load a new image
img_path = 'path_to_your_image.png'
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = preprocess_input(img_array)  # same preprocessing you used during training
img_array = tf.expand_dims(img_array, 0)  # model.predict expects a batch of images

# Make a prediction
predictions = model.predict(img_array)
x_pred, y_pred = predictions[0]
print(f"Predicted coordinates: ({x_pred}, {y_pred})")




# Save the entire model to a HDF5 file
model.save('my_model.h5')

# Later, you can load the same model
from tensorflow.keras.models import load_model
model = load_model('my_model.h5')
