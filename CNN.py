import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from tensorflow.keras.datasets import cifar10


(train_images, train_labels), (test_images, test_labels) = np.array(cifar10.load_data())
train_images, test_images = train_images / 255.0, test_images / 255.0


print("Training images shape:", train_images.shape)
print("Training labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)

model = tf.keras.Sequential()
model.add( tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add( tf.keras.layers.MaxPooling2D((2, 2)))
model.add( tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add( tf.keras.layers.MaxPooling2D((2, 2)))
model.add( tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.add( tf.keras.layers.Flatten())
model.add( tf.keras.layers.Dense(64, activation='relu'))
model.add( tf.keras.layers.Dense(10))

#odel = tf.keras.Sequential([
  #  tf.keras.layers.Dense(input_shape=(1, 3)),
 #   tf.keras.layers.Dense(1000),
 #   tf.keras.layers.Dense(11
#])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))
#0.2477
#0.3411
#0.3874

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])

predictions = probability_model.predict([test_images])

n = 30
class_names = ["airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"]

correct = 0
wrong = 0
for n in range(1000):
  if class_names[test_labels[np.argmax(predictions[n])][0]] == class_names[test_labels[n][0]]:
    correct += 1
  else:
    wrong += 1




#  plt.imshow(test_images[n], interpolation='nearest')
 # plt.axis('off')  # Turn off axis numbers and ticks
  #plt.show()
 # print(f" predicted {class_names[test_labels[np.argmax(predictions[n])][0]]} \n true {class_names[test_labels[n][0]]}")


print(f"wrong {wrong} \ncorrect {correct}")
