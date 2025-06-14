import tensorflow as tf
from tensorflow.keras import layers

class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim
        )
        
        self.layernorm = layers.LayerNormalization()
        self.add = layers.Add()
        
    def call(self, inputs, training=None):
        # Multi-head attention
        attn_output = self.mha(
            inputs, inputs,
            return_attention_scores=False,
            training=training
        )
        
        # Add & normalize
        out = self.add([inputs, attn_output])
        out = self.layernorm(out)
        
        return out
        
    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim
        })
        return config 