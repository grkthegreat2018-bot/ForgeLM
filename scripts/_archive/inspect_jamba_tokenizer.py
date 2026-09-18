"""Inspect Jamba tokenizer special tokens and tool call encoding."""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('research/checkpoints/forgelm_v2_tokenizer')

print(f'Vocab size: {tok.vocab_size}')
print(f'BOS: {tok.bos_token!r} = {tok.bos_token_id}')
print(f'EOS: {tok.eos_token!r} = {tok.eos_token_id}')
print(f'PAD: {tok.pad_token!r} = {tok.pad_token_id}')
print(f'UNK: {tok.unk_token!r} = {tok.unk_token_id}')

added = tok.added_tokens_decoder
print('\nAdded tokens (518-545):')
for tid in range(518, 545):
    if tid in added:
        print(f'  {tid}: {added[tid].content!r}')

# Check tool call tag tokenization
print('\nTool call tag tokenization:')
for s in ['<|tool_call|>', '</|tool_call|>', '<|tool|>', '</|tool|>',
          '<|im_start|>', '<|im_end|>', '<|startoftext|>',
          '<think>', '</think>']:
    ids = tok.encode(s, add_special_tokens=False)
    print(f'  {s!r:30s} -> {ids}')

# Tokenize a full tool call example
print('\nFull tool call example:')
text = '<|tool_call|>\n{"name": "test", "arguments": {"x": 1}}\n</|tool_call|>'
ids = tok.encode(text, add_special_tokens=False)
print(f'  {len(ids)} tokens: {ids[:15]}...')

# Check thinking tags
print('\nThinking tags:')
for s in ['<think>', '</think>', '<reasoning>', '</reasoning>']:
    ids = tok.encode(s, add_special_tokens=False)
    print(f'  {s!r:30s} -> {ids}')
