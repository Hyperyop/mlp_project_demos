from pathlib import Path
from prepare_data import *
import tensorflow as tf
import pickle
with open('dic.pkl', 'rb') as f:
    ae=pickle.load( f)


vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=80,
    output_sequence_length=1)
vectorize_layer.adapt(ae)
ll = 350  # maximum number of nodes
el = 850
src_path=Path("processed_data/test_set")
A=parser(create_batches(src_path),ll,el,vectorize_layer,79)
print(A[0])
