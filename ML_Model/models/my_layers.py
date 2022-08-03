import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
from keras.engine.input_spec import InputSpec


class Dense_Id(keras.layers.Dense):
    
    def build(self, input_shape):
        
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('A Dense layer can only be built with a floating-point '
                          f'dtype. Received: dtype={dtype}')

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to a Dense layer '
                           'should be defined. Found None. '
                           f'Full input shape received: {input_shape}')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight('kernel',
                                      shape=[last_dim, self.units],
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype=self.dtype,
                                      trainable=True)
        
       
        if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=[self.units + input_shape[-1],],
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        else:
            self.bias = None
        self.built = True
    
        
    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        is_ragged = isinstance(inputs, tf.RaggedTensor)
        if is_ragged:
            # In case we encounter a RaggedTensor with a fixed last dimension (last
            # dimension not ragged), we can flatten the input and restore the ragged
            # dimensions at the end.
            if tf.compat.dimension_value(inputs.shape[-1]) is None:
                raise ValueError('Dense layer only supports RaggedTensors when the '
                             'innermost dimension is non-ragged. Received: '
                             f'inputs.shape={inputs.shape}.')
            original_inputs = inputs
            if inputs.flat_values.shape.rank > 1:
                inputs = inputs.flat_values
            else:

                # Innermost partition is encoded using uniform_row_length.
                # (This is unusual, but we can handle it.)
                if inputs.shape.rank == 2:
                    inputs = inputs.to_tensor()
                    is_ragged = False
                else:
                    for _ in range(original_inputs.ragged_rank - 1):
                        inputs = inputs.values
                        inputs = inputs.to_tensor()
                        original_inputs = tf.RaggedTensor.from_nested_row_splits(inputs, original_inputs.nested_row_splits[:-1])

        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            # We use embedding_lookup_sparse as a more efficient matmul operation for
            # large sparse input tensors. The op will result in a sparse gradient, as
            # opposed to sparse_ops.sparse_tensor_dense_matmul which results in dense
            # gradients. This can lead to sigfinicant speedups, see b/171762937.
            if isinstance(inputs, tf.SparseTensor):
                # We need to fill empty rows, as the op assumes at least one id per row.
                inputs, _ = tf.sparse.fill_empty_rows(inputs, 0)
                # We need to do some munging of our input to use the embedding lookup as
                # a matrix multiply. We split our input matrix into separate ids and
                # weights tensors. The values of the ids tensor should be the column
                # indices of our input matrix and the values of the weights tensor
                # can continue to the actual matrix weights.
                # The column arrangement of ids and weights
                # will be summed over and does not matter. See the documentation for
                # sparse_ops.sparse_tensor_dense_matmul a more detailed explanation
                # of the inputs to both ops.
                ids = tf.SparseTensor(indices=inputs.indices,
                                      values=inputs.indices[:, 1],
                                      dense_shape=inputs.dense_shape)
                weights = inputs
                outputs = tf.nn.embedding_lookup_sparse(self.kernel, ids, weights, combiner='sum')
                outputs = tf.concat([outputs, inputs], -1)

            else:
                outputs = tf.matmul(a=inputs, b=self.kernel)
                outputs = tf.concat([outputs, inputs], -1)
        # Broadcast kernel to inputs.
        else:
            ###############################################
            ## ICI ! il faut faire le dot avec une partie de Kernel correspondant à la première partie de dim = units - input_last_dim
            ## Il faut ensuite concatener avec les inputs 
            ### il faut proposer la possibilité d'appliquer la fonction d'activation en deux endroit différent 
            outputs = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            outputs = tf.concat([outputs, inputs], -1)
            # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():

                shape = inputs.shape.as_list()
                #output_shape = shape[:-1] + [self.kernel.shape[-1]]
                output_shape = shape[:-1] + [outputs.shape[-1]]
                outputs.set_shape(output_shape)

        #ATENTION !!!!
        ## Le shape de bias étatt
        if self.use_bias:
            #outputs = tf.nn.bias_add(outputs, tf.concat(self.bias,tf.convert_to_tensor( tf.zeros(shape= [inputs.shape[-1],]))))
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        if is_ragged:
            outputs = original_inputs.with_flat_values(outputs)

        return outputs
    
    
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the input shape of a Dense layer '
                           'should be defined. Found None. '
                           f'Received: input_shape={input_shape}')
        return input_shape[:-1].concatenate(self.units  + inputs.shape[-1])


        


        

    
    