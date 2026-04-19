"""Prototype utterances and intent metadata for the router."""

from __future__ import annotations

from typing import Dict, List


INTENT_PROTOTYPES: Dict[str, List[str]] = {
    "chat": [
        "hi",
        "hello there",
        "thanks",
        "can you explain that more casually",
        "what do you think",
    ],
    "knowledge_qa": [
        "what is etops",
        "explain this aviation concept",
        "how does this process work",
        "what are the requirements for alternate minimums",
    ],
    "doc_qa": [
        "what does this document say",
        "who owns this task in the file",
        "what does section 3 say",
        "summarize this file",
    ],
    "summarization": [
        "summarize this",
        "give me the key points",
        "condense this document",
        "tl dr this file",
    ],
    "comparison": [
        "compare these two files",
        "what changed between v1 and v2",
        "compare policy a and b",
        "difference between these docs",
    ],
    "session_memory": [
        "what did we decide earlier",
        "continue from where we left off",
        "what was the plan again",
        "remind me what we discussed",
    ],
    "task_action": [
        "index this file",
        "delete this source",
        "re embed the corpus",
        "pin this answer",
    ],
    "fallback": [
        "this one",
        "what about that",
        "can you help",
        "not sure",
    ],
}
