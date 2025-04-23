# Utility to recursively remove a key from nested dicts/lists

def deep_remove_key(obj, key_to_remove):
    """Recursively remove all occurrences of key_to_remove from nested dicts/lists."""
    if isinstance(obj, dict):
        return {k: deep_remove_key(v, key_to_remove) for k, v in obj.items() if k != key_to_remove}
    elif isinstance(obj, list):
        return [deep_remove_key(item, key_to_remove) for item in obj]
    else:
        return obj
