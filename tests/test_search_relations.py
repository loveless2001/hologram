#!/usr/bin/env python3
"""
Test the enhanced /search endpoint with concept relations.
"""
import requests
import json

API_URL = "http://localhost:8000"

def test_search_with_relations():
    print("=" * 60)
    print("Testing Enhanced /search Endpoint with Relations")
    print("=" * 60)
    
    # 1. Load KB
    print("\n1. Loading KB...")
    response = requests.post(f"{API_URL}/chat", json={
        "message": "load",
        "kb_name": "relativity.txt"
    })
    print(f"   Status: {response.status_code}")
    
    # 2. Search with relations
    print("\n2. Searching for 'speed of light' with relations...")
    response = requests.post(f"{API_URL}/search", json={
        "query": "speed of light",
        "top_k": 3
    })
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n   Query: '{data['query']}'")
        print(f"   Found {len(data['results'])} results\n")
        
        for i, result in enumerate(data['results'], 1):
            print(f"   {i}. {result['content']}")
            print(f"      Similarity: {result['score'] * 100:.1f}%")
            
            if result.get('relations'):
                print(f"      Relations ({len(result['relations'])}):")
                for rel in result['relations'][:3]:  # Show top 3
                    print(f"         → {rel['concept'][:50]} (strength: {rel['strength'] * 100:.1f}%)")
            else:
                print("      Relations: None")
            print()
    else:
        print(f"   Error: {response.status_code}")
        print(f"   {response.text}")
    
    # 3. Search for a different keyword
    print("\n3. Searching for 'time dilation' with relations...")
    response = requests.post(f"{API_URL}/search", json={
        "query": "time dilation",
        "top_k": 3
    })
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n   Query: '{data['query']}'")
        print(f"   Found {len(data['results'])} results\n")
        
        for i, result in enumerate(data['results'], 1):
            print(f"   {i}. {result['content']}")
            print(f"      Similarity: {result['score'] * 100:.1f}%")
            
            if result.get('relations'):
                print(f"      Top Relations:")
                for rel in result['relations'][:3]:
                    print(f"         → {rel['concept'][:50]} (strength: {rel['strength'] * 100:.1f}%)")
            print()
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_search_with_relations()
