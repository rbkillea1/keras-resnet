from __future__ import absolute_import

from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Wrapper, Merge
from keras import regularizers, constraints, initializations

class ResMerge(Merge):
    def __init__(self, dims, p=None,
                 concat_axis=-1,
                 dot_axes=-1, output_shape=None, output_mask=None,
                 arguments=None, node_indices=None, tensor_indices=None,
                 name=None, **kwargs):
        super(Merge, self).__init__(**kwargs)
        self.mode=self.resmerge
        self.concat_axis = concat_axis
        self.dot_axes = dot_axes
        self._output_shape = output_shape
        self.node_indices = node_indices
        self._output_mask = output_mask
        self.arguments = arguments if arguments else {}

        # Layer parameters.
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.constraints = {}
        self._trainable_weights = []
        self._non_trainable_weights = []
        self.supports_masking = True
        self.uses_learning_phase = False
        self.input_spec = None  # Compatible with anything.
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))
        self.dims = dims
        if not p:
            self.p = self.add_weight((1,),
                                     initializer='uniform',
                                     name='{}_p'.format(self.name),
                                     regularizer=regularizers.get(None),
                                     constraint=constraints.get(None))
        self.arguments = {}

    def resmerge(self, inputs, **kwargs):
        x, y = inputs[0], inputs[1]
        ptrue = K.sqrt(K.sigmoid(self.p))
        split = K.tile(ptrue, self.dims)
        resout = split*x + (1-split)*y
        sample = K.tile(K.in_train_phase(K.random_binomial((1,), p=ptrue), K.zeros((1,))), self.dims)
        output = sample*x + (1-sample)*resout
        return output

class Residual(Wrapper):
    """This wrapper automatically applies a (modified) residual to a model.

    For an input `x` and a model `F(x)`, the residual wrapper gives the output
    `y = sqrt(p)*F(x) + (1-sqrt(p))*x` with probability sqrt(p).
    The rationale for this modification is that it combines sampling x ~ X and E[X] in
    a hypothetical model where the inclusion of a layer is done via a bernoulli RV.
    In this configuration, the output of F(x) must have the
    same shape as x. I got rid of the merge mode aspect of this as it doesn't make sense
    with our reformulation.

    Arguments:
        layer: The layer to wrap
        p_regularizer: the regularizer for updating p
        p_constraint: the constraint for updating p
    """
    def __init__(self, layer, p=None,
                 **kwargs):
        self.layer = layer
        self.p = p
        super(Residual, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        output_shape = self.layer.get_output_shape_for(input_shape)
        if output_shape != input_shape:
            raise Exception('Cannot apply residual to layer "{}": '
                            'mismatching input and output shapes'
                            '"{}" and "{}"'
                            .format(self.layer.name, input_shape, output_shape))
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        self.input_spec = [InputSpec(shape=input_shape)]
        super(Residual, self).build()
        self.res = None
        
    def get_output_shape_for(self, input_shape):
        return input_shape
    
    def call(self, x, mask=None):
        layer_output = self.layer.call(x, mask)
        if not self.res:
            self.res = ResMerge(dims=self.layer.get_output_shape_for(x.shape), p=self.p)
        return self.res([layer_output, x])
    
    @classmethod
    def from_config(cls, config):
        from keras.utils.layer_utils import layer_from_config
        self.res = layer_from_config(config.pop('res'))
        self.p = res.p
        residual = super(Residual, cls).from_config(config)
        return residual
    
    def get_config(self):
        config = {"ResMerge": {'class_name': 'ResMerge',
                               'config': self.res.get_config()}}
        base_config = super(Residual, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

