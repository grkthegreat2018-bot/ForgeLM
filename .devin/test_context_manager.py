"""Test context manager with a long conversation that triggers compression."""
import sys
sys.path.insert(0, r"D:\windsurf\ForgeAI")

from research.self_play.discovery.context_manager import (
    ContextManager, ContextManagerConfig, count_conversation_tokens,
    build_summary_message, split_conversation,
)

# Build a long conversation (many tool calls + results)
msgs = [{"role": "user", "content": "Research Python async programming and summarize your findings."}]

# Simulate 15 turns of tool calls + results
for i in range(15):
    msgs.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{"name": "web_search", "arguments": {"query": f"python async topic {i}"}}],
    })
    msgs.append({
        "role": "tool",
        "name": "web_search",
        "content": f'{{"results": [{{"title": "Async Guide Part {i}", "snippet": "Python asyncio is a library for writing concurrent code using async/await syntax. It is used for IO-bound and high-level structured network code."}}]}}',
    })
    msgs.append({
        "role": "assistant",
        "content": f"I found that async topic {i} relates to asyncio patterns. Let me search more.",
    })

msgs.append({"role": "assistant", "content": "Python async programming uses asyncio for concurrent IO operations."})

print(f"Original: {len(msgs)} messages, {count_conversation_tokens(msgs)} tokens")

# Create context manager with small budget to force compression
cm = ContextManager(ContextManagerConfig(
    max_seq_len=500,  # tiny budget to force compression
    reserved_for_generation=100,
    keep_recent_turns=6,
))

new_msgs, compressed = cm.maybe_compress(msgs)
new_tokens = count_conversation_tokens(new_msgs)

print(f"Compressed: {compressed}")
print(f"New: {len(new_msgs)} messages, {new_tokens} tokens")
print(f"Summaries made: {cm._summary_count}")
print(f"\nFirst 3 messages of compressed:")
for m in new_msgs[:3]:
    content = m.get("content", "")
    if content and len(content) > 100:
        content = content[:100] + "..."
    print(f"  [{m['role']}] {content}")

print("\nOK")
