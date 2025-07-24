import tensorflow as tf #type: ignore

if(tf.config.list_physical_devices("GPU")):
    print("Tensorflow has access to the GPU")
else:
    print("Tensorflow does not has access to the GPU")

print("\n", tf.config.list_physical_devices("GPU"))