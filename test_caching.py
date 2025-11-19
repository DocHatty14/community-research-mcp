import os
import json
import time
from pathlib import Path
from community_research_mcp import set_cached_result, get_cached_result, load_cache, save_cache, CACHE_FILE

def test_persistent_caching():
    print("Testing Persistent Caching...")
    
    # Clean up existing cache
    if CACHE_FILE.exists():
        os.remove(CACHE_FILE)
    
    # Set a value
    key = "test_key"
    value = "test_value"
    set_cached_result(key, value)
    
    # Verify in memory
    assert get_cached_result(key) == value
    print("✓ In-memory cache working")
    
    # Verify on disk
    assert CACHE_FILE.exists()
    content = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    assert key in content
    assert content[key]["result"] == value
    print("✓ Disk persistence working")
    
    # Simulate restart (reload from disk)
    # Manually clear global cache to simulate restart
    import community_research_mcp
    community_research_mcp._cache = {}
    
    # Load from disk
    community_research_mcp._cache = load_cache()
    assert get_cached_result(key) == value
    print("✓ Cache reload working")
    
    print("✅ Persistent Caching Verified!")

if __name__ == "__main__":
    test_persistent_caching()
