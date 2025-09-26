import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.clustering.keras import cluster_weights
from tensorflow_model_optimization.python.core.clustering.keras import CentroidInitialization

# Load trained model
model = tf.keras.models.load_model('model/final_model.h5')

# Pruning
prune_low_magnitude = sparsity.prune_low_magnitude
pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.0,
                                                 final_sparsity=0.5,
                                                 begin_step=0,
                                                 end_step=1000)
}
model_pruned = prune_low_magnitude(model, **pruning_params)
model_pruned.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Clustering
model_clustered = cluster_weights(model_pruned, number_of_clusters=16, cluster_centroids_init=CentroidInitialization.KMEANS_PLUS_PLUS)
model_clustered.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Quantization and conversion to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model_clustered)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('model/breed_predictor.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model optimized and converted to TensorFlow Lite.")
