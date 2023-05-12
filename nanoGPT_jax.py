from dataclasses import dataclass
from functools import partial
import pickle

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.training import train_state
from flax import serialization

import optax


@dataclass
class Config():
    seed = 42
    num_iterations = 20000
    batch_size = 512
    block_size = 64
    learning_rate = 1e-4
    embed_size = 256
    num_heads = 8
    head_size = 32
    num_layers = 6
    dropout = 0.2

config = Config()

with open("inputs/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: "".join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Let's now split up the data into train and validation sets
data = jnp.array(encode(text))
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
eval_data = data[n:]

dynamic_slice_vmap = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))

@jax.jit
def get_batch(random_key, data):
    # generate a small batch of data of inputs x and targets y
    ix = jax.random.randint(random_key, shape=(config.batch_size, 1), minval=0, maxval=len(data)-config.block_size)
    x = dynamic_slice_vmap(data, ix, (config.block_size,))
    y = dynamic_slice_vmap(data, ix+1, (config.block_size,))
    return x, y

class LayerNorm(nn.Module):
    epsilon: float = 1e-6
    reduction_axes = -1

    @nn.compact
    def __call__(self, x):
        """Applies layer normalization on the input."""
        # compute statistics
        mean2 = jnp.mean(jax.lax.square(x), self.reduction_axes, keepdims=True)
        mean = jnp.mean(x, self.reduction_axes, keepdims=True)
        var = jnp.maximum(0., mean2 - jax.lax.square(mean))

        # compute normalized inputs
        x_norm = (x - mean) * jax.lax.rsqrt(var + self.epsilon)
        return x_norm * self.param("scale", nn.initializers.ones, x.shape[-1]) + self.param("bias", nn.initializers.zeros, x.shape[-1])

class Attention(nn.Module):
    head_size: int

    @nn.compact
    def __call__(self, x, training: bool):
        key = nn.Dense(self.head_size, use_bias=False)(x)
        query = nn.Dense(self.head_size, use_bias=False)(x)
        value = nn.Dense(self.head_size, use_bias=False)(x)
        
        tril = jnp.tril(jnp.ones((x.shape[-2], x.shape[-2])))
        attention_weights = nn.softmax(jnp.where(tril == 0, -jnp.inf, query @ jnp.transpose(key, axes=(0, 2, 1))), axis=-1)
        attention_weights = nn.Dropout(config.dropout)(attention_weights, deterministic=not training)
        return attention_weights @ value

class MultiHeadAttention(nn.Module):
    num_heads: int
    head_size: int

    @nn.compact
    def __call__(self, x, training: bool):
        x = jnp.concatenate([Attention(self.head_size)(x, training) for _ in range(self.num_heads)], axis=-1)
        return nn.Dropout(config.dropout)(nn.Dense(self.num_heads*self.head_size)(x), deterministic=not training)

class FeedFoward(nn.Module):

    @nn.compact
    def __call__(self, x, training: bool):
        return nn.Dropout(config.dropout)(nn.Dense(config.embed_size)(nn.relu(nn.Dense(4*config.embed_size)(x))), deterministic=not training)

class Block(nn.Module):
    num_heads: int
    head_size: int

    @nn.compact
    def __call__(self, x, training: bool):
        x = x + MultiHeadAttention(self.num_heads, self.head_size)(LayerNorm()(x), training)
        return x + FeedFoward()(LayerNorm()(x), training)

class Model(nn.Module):
    num_layers: int
    num_heads: int
    head_size: int

    @nn.compact
    def __call__(self, x, training: bool):
        B, T = x.shape
        x = nn.Embed(num_embeddings=vocab_size, features=config.embed_size)(x) + \
            nn.Embed(num_embeddings=config.block_size, features=config.embed_size)(jnp.arange(T))
        for _ in range(self.num_layers):
            x = Block(self.num_heads, self.head_size)(x, training)
        x = nn.LayerNorm()(x)
        return nn.Dense(vocab_size)(x)

    def generate(self, random_key, params, context, length=50):
        for _ in range(length):
            logits = self.apply(params, context[:, -config.block_size:], training=False)
            random_key, random_subkey = jax.random.split(random_key)
            new_token = jax.random.categorical(random_subkey, logits[:, -1, :], axis=-1, shape=(1, 1))
            context = jnp.concatenate([context, new_token], axis=1)
        return context

    @partial(jax.jit, static_argnames=("self", "length"))
    def generate_jit(self, random_key, params, length):
        def scan_generate(carry, x):
            key, context = carry
            logits = self.apply(params, context, training=False)
            random_key, random_subkey = jax.random.split(key)
            new_token = jax.random.categorical(random_subkey, logits[:, -1, :], axis=-1, shape=(1, 1))
            context = jnp.concatenate([context[:, 1:], new_token], axis=1)
            return (random_key, context), new_token
        
        _, new_tokens = jax.lax.scan(
            scan_generate,
            (random_key, jnp.zeros((1, config.block_size), dtype=jnp.int32)),
            (),
            length=length,
        )
        return new_tokens

class TrainState(train_state.TrainState):
  key: jax.random.KeyArray

def create_train_state(random_key, config):
    model = Model(num_layers=config.num_layers, num_heads=config.num_heads, head_size=config.head_size)
    params = model.init(random_key, jnp.ones((config.batch_size, config.block_size), dtype=jnp.int32), training=False)
    tx = optax.adamw(config.learning_rate)
    return TrainState.create(
        apply_fn=model.apply, params=params, key=random_key, tx=tx)

@jax.jit
def train_step(state, x, y, dropout_key):
    dropout_key = jax.random.fold_in(key=dropout_key, data=state.step)
    def loss_fn(params):
        logits = state.apply_fn(params, x, training=True, rngs={'dropout': dropout_key})
        one_hot_encoded_labels = jax.nn.one_hot(y, num_classes=vocab_size)
        return optax.softmax_cross_entropy(
            logits=logits, labels=one_hot_encoded_labels
        ).mean()
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss

@jax.jit
def eval_step(state, x, y):
    logits = state.apply_fn(state.params, x, training=False)
    one_hot_encoded_labels = jax.nn.one_hot(y, num_classes=vocab_size)
    return optax.softmax_cross_entropy(
        logits=logits, labels=one_hot_encoded_labels
    ).mean()

random_key = jax.random.PRNGKey(config.seed)
random_key, random_subkey = jax.random.split(random_key)

state = create_train_state(random_subkey, config)
for i in range(config.num_iterations):
    random_key, random_subkey = jax.random.split(random_key)
    state, loss = train_step(state, *get_batch(random_subkey, train_data), random_subkey)
    
    if i % 100 == 0:
        random_key, random_subkey = jax.random.split(random_key)
        print(f"Step: {i}\t train loss: {loss}\t eval loss: {eval_step(state, *get_batch(random_subkey, eval_data))}")

params_state_dict = serialization.to_state_dict(state.params)
with open("./outputs/params.pickle", "wb") as params_file:
    pickle.dump(params_state_dict, params_file)

# Let's now generate some text
model = Model(num_layers=config.num_layers, num_heads=config.num_heads, head_size=config.head_size)
params = model.init(
    random_key, jnp.ones((config.batch_size, config.block_size), dtype=jnp.int32), training=False
)
with open("./outputs/params.pickle", "rb") as params_file:
    params_state_dict = pickle.load(params_file)
params = serialization.from_state_dict(params, params_state_dict)

text = model.generate_jit(random_key, params, 1000)[:, 0, 0].tolist()
print(decode(text))
