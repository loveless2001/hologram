
import os
from pathlib import Path
from hologram.api import Hologram
from hologram.chatbot import ChatMemory, ChatWindow, SessionLog, resolve_provider

def run_conversation():
    # Initialize Hologram (using hashing for speed/simplicity in this test, or CLIP if available)
    # Note: Using use_clip=False to avoid dependency issues if not set up, but real usage might prefer True
    hologram = Hologram.init(use_clip=False, use_gravity=True)
    
    # Initialize Chat components
    memory = ChatMemory(hologram=hologram)
    logs = SessionLog(Path("data/chat_logs"))
    
    # Resolve provider (will load from .env and use GPT-5)
    provider = resolve_provider()
    
    # Create ChatWindow
    session_id = "gpt5_test_session_v2"
    window = ChatWindow(
        provider=provider,
        memory=memory,
        logs=logs,
        session_id=session_id,
        system_prompt="You are a helpful AI assistant connected to a holographic memory system."
    )
    
    print(f"--- Starting conversation with {provider.__class__.__name__} ---")
    
    # 1. Ask a complex abstract question
    q1 = "Explain the concept of 'time' from the perspective of a photon. How does it relate to the speed of light?"
    print(f"\nUser: {q1}")
    reply1 = window.step(q1)
    print(f"Bot: {reply1}")
    
    # 2. Ask a follow-up about memory/entropy
    q2 = "Now connect that to the concept of entropy and the arrow of time. Does a photon experience entropy?"
    print(f"\nUser: {q2}")
    reply2 = window.step(q2)
    print(f"Bot: {reply2}")
    
    # 3. Ask for a synthesis that requires recalling the first answer
    q3 = "Synthesize your previous two answers into a haiku about light and time."
    print(f"\nUser: {q3}")
    reply3 = window.step(q3)
    print(f"Bot: {reply3}")
    
    # 4. Verify memory storage directly
    print("\n--- Verifying Memory Store ---")
    recent_msgs = memory.get_recent_session_messages(session_id)
    print(f"Stored {len(recent_msgs)} messages in session context.")
    
    # Search for "photon" in the global context
    print("\n--- Searching Holographic Memory for 'photon' ---")
    hits = memory.search_global_context("photon", top_k=3)
    for hit in hits:
        print(f"Found trace: [{hit.get('role')}] {hit.get('content')[:100]}...")

    # 5. Verify Gravity Field Concepts (Decomposition Check)
    print("\n--- Verifying Gravity Field Decomposition ---")
    if hologram.field:
        concepts = hologram.field.sim.concepts
        print(f"Total concepts in gravity field: {len(concepts)}")
        
        # Check for specific expected concepts
        expected = ["photon", "time", "entropy", "light", "speed"]
        found = []
        for term in expected:
            # We check if any concept name contains the term (case insensitive)
            matches = [c for c in concepts.keys() if term in c.lower()]
            if matches:
                found.append(f"{term} -> {matches[:3]}")
        
        if found:
            print("Found decomposed concepts:")
            for f in found:
                print(f"  - {f}")
        else:
            print("No expected concepts found in gravity field.")
    else:
        print("Gravity field not active.")

if __name__ == "__main__":
    run_conversation()
