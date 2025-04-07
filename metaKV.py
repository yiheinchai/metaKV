import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Model, GPT2LMHeadModel, GPT2Config
from transformers.utils import logging
import math
from typing import Optional, Tuple, Union

logger = logging.get_logger(__name__)

class MetaGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        # --- Meta-KV Cache Specific Parameters ---
        self.meta_rank = config.meta_rank if hasattr(config, "meta_rank") else 4 # Rank for low-rank adaptation
        self.meta_context_pooling = config.meta_context_pooling if hasattr(config, "meta_context_pooling") else "mean" # Could be 'mean', 'max', 'attention'

        # Layers to generate the low-rank updates based on meta context
        # Input dimension is embed_dim (from pooled hidden states)
        # Output dimension needs to be rank * embed_dim for A and embed_dim * rank for B (transposed)
        # We generate vectors that are reshaped

        # Using nn.Linear for simplicity to generate parameters for A and B
        # Project pooled context to a space that generates the parameters for A*B
        self.meta_state_dim = self.embed_dim # Dimension of the derived meta state
        projection_size = self.meta_rank * (self.embed_dim + self.head_dim * self.num_heads) # Size needed for A_k, B_k, A_v, B_v factors

        # Simplified: Project meta-state to generate parameters for low-rank matrices
        # We need parameters for: A_k (embed_dim, rank), B_k (rank, embed_dim) -> (embed_dim * rank) + (rank * embed_dim) params
        # And              for: A_v (embed_dim, rank), B_v (rank, embed_dim) -> (embed_dim * rank) + (rank * embed_dim) params
        # Let's generate the factors directly for simplicity in prototype
        self.meta_k_a_proj = nn.Linear(self.meta_state_dim, self.embed_dim * self.meta_rank)
        self.meta_k_b_proj = nn.Linear(self.meta_state_dim, self.meta_rank * self.embed_dim)
        self.meta_v_a_proj = nn.Linear(self.meta_state_dim, self.embed_dim * self.meta_rank)
        self.meta_v_b_proj = nn.Linear(self.meta_state_dim, self.meta_rank * self.embed_dim)

        # Initialize the new layers
        # Consider specific initialization strategies (e.g., zero init for B projections)
        nn.init.normal_(self.meta_k_a_proj.weight, std=0.02)
        nn.init.zeros_(self.meta_k_a_proj.bias)
        nn.init.zeros_(self.meta_k_b_proj.weight) # Important: Init B to zero so initial delta is zero
        nn.init.zeros_(self.meta_k_b_proj.bias)
        nn.init.normal_(self.meta_v_a_proj.weight, std=0.02)
        nn.init.zeros_(self.meta_v_a_proj.bias)
        nn.init.zeros_(self.meta_v_b_proj.weight) # Important: Init B to zero
        nn.init.zeros_(self.meta_v_b_proj.bias)
        # --- End Meta-KV Cache Specific Parameters ---


    def _get_meta_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Computes the meta state from hidden states.
        Input: hidden_states (batch_size, seq_len, embed_dim)
        Output: meta_state (batch_size, embed_dim) - pooled representation
        """
        # Simple mean pooling across sequence dimension for demonstration
        # In a real scenario, might want causal pooling or pooling only over specific past context
        if self.meta_context_pooling == "mean":
            # Pool across the sequence length dimension
            # For causal, you'd pool hidden_states[:, :current_token_idx, :]
            # For simplicity here, pool across all available sequence length
            # Note: In generation, seq_len grows. A causal pool is better.
            # For this prototype, let's just pool the whole input batch for simplicity
            meta_state = torch.mean(hidden_states, dim=1) # (batch_size, embed_dim)
        else:
            # Implement other pooling like max or attention-based pooling here
            raise NotImplementedError(f"Pooling type '{self.meta_context_pooling}' not implemented.")

        # Detach? Maybe not during training. Depends if meta-state gradient should flow back.
        # Let's assume it flows back for end-to-end training.
        return meta_state


    def _get_kv_projections(self, hidden_states, meta_state):
        """Calculates K and V projections including meta-kv modulation."""
        batch_size, seq_len, _ = hidden_states.size()

        # Original projections (weights are from self.c_attn, split later)
        # self.c_attn combines Q, K, V projections: output is (batch, seq_len, 3 * embed_dim)
        query_key_value = self.c_attn(hidden_states)
        query, key, value = query_key_value.split(self.split_size, dim=2)

        # --- Meta-KV Modification ---
        # Generate low-rank factors A and B from the meta_state
        # meta_state shape: (batch_size, embed_dim)

        # Unsqueeze meta_state to make projections work per batch item: (batch_size, 1, embed_dim)
        # However, linear layers expect (batch, features), so we process the batch
        meta_k_a_flat = self.meta_k_a_proj(meta_state) # (batch_size, embed_dim * rank)
        meta_k_b_flat = self.meta_k_b_proj(meta_state) # (batch_size, rank * embed_dim)
        meta_v_a_flat = self.meta_v_a_proj(meta_state) # (batch_size, embed_dim * rank)
        meta_v_b_flat = self.meta_v_b_proj(meta_state) # (batch_size, rank * embed_dim)

        # Reshape factors A and B
        # A: (batch_size, embed_dim, rank)
        # B: (batch_size, rank, embed_dim)
        meta_k_a = meta_k_a_flat.view(batch_size, self.embed_dim, self.meta_rank)
        meta_k_b = meta_k_b_flat.view(batch_size, self.meta_rank, self.embed_dim)
        meta_v_a = meta_v_a_flat.view(batch_size, self.embed_dim, self.meta_rank)
        meta_v_b = meta_v_b_flat.view(batch_size, self.meta_rank, self.embed_dim)

        # Calculate low-rank updates delta_Wk = A_k @ B_k and delta_Wv = A_v @ B_v
        # delta_Wk shape: (batch_size, embed_dim, embed_dim)
        # delta_Wv shape: (batch_size, embed_dim, embed_dim)
        delta_Wk = torch.bmm(meta_k_a, meta_k_b)
        delta_Wv = torch.bmm(meta_v_a, meta_v_b)

        # Apply the delta to the K and V projections
        # Need to apply batch-wise matrix multiplication:
        # hidden_states @ (W_k + delta_Wk)
        # Original Wk, Wv are slices of self.c_attn.weight. Need careful handling.

        # Easier approach: Modify K and V *after* initial projection
        # delta_K = hidden_states @ delta_Wk
        # delta_V = hidden_states @ delta_Wv
        # Use torch.bmm for batch matrix multiplication:
        # hidden_states: (batch, seq_len, embed_dim) -> permute to (batch, embed_dim, seq_len)
        # delta_Wk: (batch, embed_dim, embed_dim)
        # Result: (batch, embed_dim, seq_len) -> permute back to (batch, seq_len, embed_dim)

        # Note: hidden_states may vary in sequence length during generation.
        # meta_state is calculated once based on available context. delta_W is fixed for this step.
        # Applying delta_W to hidden_states:
        # Need hidden_states @ delta_W.T but dimensions are tricky with batching.
        # Let's compute delta_K = (hidden_states @ delta_Wk_transpose) - simpler matmul
        # delta_K = torch.bmm(hidden_states, delta_Wk.transpose(1, 2)) # No, this isn't right either.

        # Simplest conceptual change: Modify W_k and W_v before projection.
        # This is hard because W_k, W_v are part of c_attn.weight.
        # Let's approximate by adding delta *after* projection:
        # K_mod = K + (hidden_states @ delta_Wk)  <- How to batch this efficiently?

        # Alternative: Add the modification directly to K and V
        # delta_K = torch.einsum('bsd,bde->bse', hidden_states, delta_Wk) # If shapes align
        # delta_V = torch.einsum('bsd,bde->bse', hidden_states, delta_Wv)
        # This computes the effect of delta_W applied to hidden_states

        # Let's try the most direct modification to K, V:
        # Assume delta_Wk/v are *added* to the original weights.
        # The resulting K, V would be:
        # K = hidden_states @ Wk + hidden_states @ delta_Wk
        # V = hidden_states @ Wv + hidden_states @ delta_Wv
        # So, we compute the second term and add it.

        # Need efficient batch matmul: (batch, seq_len, embed_dim) @ (batch, embed_dim, embed_dim)
        # -> (batch, seq_len, embed_dim)
        delta_k_per_token = torch.bmm(hidden_states, delta_Wk)
        delta_v_per_token = torch.bmm(hidden_states, delta_Wv)

        # Add the delta adjustment
        # Scaling factor might be needed (like in LoRA) e.g., alpha / rank
        scaling = 1.0 / self.meta_rank # Example scaling
        key = key + delta_k_per_token * scaling
        value = value + delta_v_per_token * scaling
        # --- End Meta-KV Modification ---

        return query, key, value

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        # --- Meta-KV State Calculation ---
        # Calculate meta state based on current hidden states
        # Note: In generation (use_cache=True), hidden_states might just be the *last* token's state.
        # We need access to the *full* sequence history ideally. This complicates things.
        # Workaround for prototype: If use_cache, maybe get context from layer_past?
        # For now, assume hidden_states contains relevant context (e.g., during training or first pass)
        meta_state = self._get_meta_state(hidden_states)
        # --- End Meta-KV State Calculation ---

        if encoder_hidden_states is not None:
            # Cross-Attention Implementation (Standard)
            # ... (standard cross-attention logic using K/V from encoder)
            # Meta-KV usually applies to self-attention, maybe skip modification here?
            # For now, let's assume Meta-KV only applies to self-attention
            if not self.is_cross_attention:
                 raise ValueError("Meta-KV is designed for self-attention here.")
            # Standard cross-attn projection
            query = self.q_attn(hidden_states)
            key, value = self.kv_attn(encoder_hidden_states).split(self.split_size, dim=2)

        else:
            # Self-Attention Implementation (Modified)
            # Calculate Q, K_mod, V_mod using meta state influence
            query, key, value = self._get_kv_projections(hidden_states, meta_state)


        # --- Standard GPT2 Attention Logic ---
        # Split heads, transpose, handle KV cache, compute scores, apply mask, softmax, weighted sum, combine heads...
        # (The rest of the function is largely standard, but uses the *modified* K and V)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            # The `key` and `value` computed above correspond ONLY to the NEW tokens
            # layer_past contains K, V for previous tokens
            # We append the *modified* new K, V to the past K, V
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            # Store the concatenated, potentially modified, K and V
            present = (key, value)
        else:
            present = None

        # --- Standard Attention Calculation ---
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs # type: ignore


# --- Function to Replace Layers ---
def replace_gpt2_attention_with_meta(model: nn.Module, config):
    for name, module in model.named_children():
        if isinstance(module, GPT2Attention):
            # Replace the layer!
            new_layer = MetaGPT2Attention(config, module.is_cross_attention, module.layer_idx)
            # Copy pre-trained weights (c_attn, c_proj)
            new_layer.load_state_dict(module.state_dict(), strict=False) # Strict=False needed for new meta params
            setattr(model, name, new_layer)
            logger.info(f"Replaced {name} with MetaGPT2Attention.")
        elif len(list(module.children())) > 0:
            # Recursively replace in submodules
            replace_gpt2_attention_with_meta(module, config)

# --- Example Usage ---
if __name__ == '__main__':
    # 1. Load Pretrained GPT-2 Model and Tokenizer
    model_name = 'gpt2'
    config = GPT2Config.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # 2. Add Meta-KV specific config (optional, or pass directly)
    config.meta_rank = 8 # Example rank
    config.meta_context_pooling = 'mean'

    # 3. Replace Attention Layers
    # Need to iterate through the model structure (e.g., model.transformer.h is the list of blocks)
    for i, layer in enumerate(model.transformer.h):
         # Create the new layer with potentially unique layer_idx if needed
         meta_attn = MetaGPT2Attention(config, is_cross_attention=False, layer_idx=i)
         # Copy weights from original attention layer's c_attn and c_proj
         orig_attn = layer.attn
         meta_attn.c_attn.load_state_dict(orig_attn.c_attn.state_dict())
         meta_attn.c_proj.load_state_dict(orig_attn.c_proj.state_dict())
         # Assign the new layer
         layer.attn = meta_attn
         print(f"Replaced attention in layer {i}")


    # 4. Model is Ready for Training/Fine-tuning
    print("Model modification complete. Ready for training.")
    model.train() # Set model to training mode

    # Example forward pass (requires tokenizer and input formatting)
    # inputs = tokenizer("Hello, how are you?", return_tensors="pt")
    # outputs = model(**inputs, labels=inputs["input_ids"])
    # loss = outputs.loss
    # print("Loss:", loss.item())

    # --- Notes on Generation/KV Caching ---
    # The handling of `layer_past` and `hidden_states` needs care.
    # When `use_cache=True`, `hidden_states` is usually just the *last* token's embedding.
    # The `_get_meta_state` function needs access to a representation of the *entire context* up to that point.
    # Options:
    #    a) Pass *all* previous hidden states (memory intensive).
    #    b) Compute meta_state based on the K/V cache contents ( `layer_past`). This seems promising.
    #    c) Maintain a separate running state representing the context (e.g., an RNN state summarizing history).
    # The current prototype's `_get_meta_state` using mean pooling on input `hidden_states`
    # works well during training but is inaccurate during cached generation.
    # A robust implementation would likely need to modify the generation loop or how context is passed.
    # For example, compute meta_state based on `key` and `value` from `layer_past` before concatenation.
