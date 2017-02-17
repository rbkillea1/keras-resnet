from __future__ import absolute_import

from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Wrapper, Merge
from keras import regularizers, constraints


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
    def __init__(self, layer, merge_mode='sum',
                 p_regularizer=None, p_constraint=None,
                 **kwargs):
        self.supports_masking = True
        self.p_regularizer=regularizers.get(p_regularizer)
        self.p_constraint=regularizers.get(p_constraint)
        layer.add_weight((1,), # 1 new parameter (p)
                         initializer=layer.init, # make it be initialized according to the same method as the layer
                         name='{}_W_presnet'.format(layer.name), # make it be named the same as the layer but with resnet
                         regularizer=self.p_regularizer,
                         constraint=self.p_constraint)
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

    def get_output_shape_for(self, input_shape):
        return input_shape

    def call(self, x, mask=None, seed=None):
        layer_output = self.layer.call(x, mask)
        ptrue = K.sqrt(K.sigmoid(p))
        resout = (1-ptrue)*layer_output + ptrue*x 
        output = K.in_train_phase(K.switch(K.random_binomial((1,), p=ptrue, seed=seed), x, resout), resout)
        return output
    
    @classmethod
    def from_config(cls, config):
        from keras.utils.layer_utils import layer_from_config
        merge_mode = layer_from_config(config.pop('merge_mode'))
        residual = super(Residual, cls).from_config(config)
        residual.merge_mode = merge_mode
        return residual
    
    def get_config(self):
        config = {"merge_mode": {'class_name': 'Merge',
                                 'config': self.merge_mode.get_config()}}
        base_config = super(Residual, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

