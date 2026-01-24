"""
Redis caching utilities for Teams-BOT.

Provides decorators for caching function results in Redis with TTL.
Falls back gracefully when Redis is unavailable.
"""
import json
import hashlib
from functools import wraps
from typing import Optional, Callable, Any, TypeVar
import asyncio

F = TypeVar('F', bound=Callable[..., Any])


def make_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate a cache key from function arguments.

    Args:
        prefix: Key prefix (usually function name)
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        A cache key string
    """
    key_parts = [prefix]

    # Add positional args
    for arg in args:
        if hasattr(arg, '__dict__'):
            # Skip self/cls parameters
            continue
        key_parts.append(str(arg)[:100])  # Limit length

    # Add sorted kwargs
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={str(v)[:100]}")

    key_string = ":".join(key_parts)

    # Hash long keys to keep them reasonable
    if len(key_string) > 200:
        hash_suffix = hashlib.md5(key_string.encode()).hexdigest()[:16]
        key_string = f"{prefix}:{hash_suffix}"

    return f"cache:{key_string}"


def cached(ttl: int = 300, key_prefix: str = None):
    """
    Decorator for caching async function results in Redis.

    Requires the instance to have a `redis_client` attribute.
    Falls back to no caching if Redis is unavailable.

    Args:
        ttl: Time-to-live in seconds (default 5 minutes)
        key_prefix: Cache key prefix (default: function name)

    Usage:
        class MyService:
            def __init__(self, redis_client):
                self.redis_client = redis_client

            @cached(ttl=600, key_prefix="equipment_count")
            async def get_equipment_count(self, category=None):
                # Expensive operation
                ...
    """
    def decorator(func: F) -> F:
        prefix = key_prefix or func.__name__

        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            # Get Redis client from self if available
            redis_client = getattr(self, 'redis_client', None)
            if not redis_client:
                # No caching available, call function directly
                return await func(self, *args, **kwargs)

            # Generate cache key
            cache_key = make_cache_key(prefix, *args, **kwargs)

            # Try to get from cache
            try:
                cached_value = await redis_client.get(cache_key)
                if cached_value:
                    return json.loads(cached_value)
            except Exception as e:
                # Cache miss or error, proceed to function
                pass

            # Call the actual function
            result = await func(self, *args, **kwargs)

            # Store in cache (fire and forget)
            try:
                serialized = json.dumps(result, default=str)
                await redis_client.setex(cache_key, ttl, serialized)
            except Exception as e:
                # Cache store failed, but we have the result
                pass

            return result

        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            # For sync functions, no async caching - just call directly
            return func(self, *args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def cached_method(ttl: int = 300, key_prefix: str = None, redis_attr: str = "redis_client"):
    """
    Alternative decorator that allows specifying the Redis client attribute name.

    Args:
        ttl: Time-to-live in seconds
        key_prefix: Cache key prefix
        redis_attr: Name of the Redis client attribute on self

    Usage:
        @cached_method(ttl=60, redis_attr="_redis")
        async def get_data(self):
            ...
    """
    def decorator(func: F) -> F:
        prefix = key_prefix or func.__name__

        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            redis_client = getattr(self, redis_attr, None)
            if not redis_client:
                return await func(self, *args, **kwargs)

            cache_key = make_cache_key(prefix, *args, **kwargs)

            try:
                cached_value = await redis_client.get(cache_key)
                if cached_value:
                    return json.loads(cached_value)
            except Exception:
                pass

            result = await func(self, *args, **kwargs)

            try:
                await redis_client.setex(cache_key, ttl, json.dumps(result, default=str))
            except Exception:
                pass

            return result

        return wrapper

    return decorator


class CacheManager:
    """
    Manager for cache operations.

    Provides methods to invalidate cache entries and manage cache state.
    """

    def __init__(self, redis_client=None):
        """
        Initialize the cache manager.

        Args:
            redis_client: Async Redis client instance
        """
        self.redis_client = redis_client

    async def invalidate(self, pattern: str) -> int:
        """
        Invalidate cache entries matching a pattern.

        Args:
            pattern: Pattern to match (e.g., "equipment_*")

        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0

        try:
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor, match=f"cache:{pattern}*", count=100
                )
                if keys:
                    deleted += await self.redis_client.delete(*keys)
                if cursor == 0:
                    break
            return deleted
        except Exception as e:
            print(f"[Cache] Invalidation error: {e}")
            return 0

    async def invalidate_prefix(self, prefix: str) -> int:
        """
        Invalidate all cache entries with a specific prefix.

        Args:
            prefix: Key prefix to invalidate

        Returns:
            Number of keys deleted
        """
        return await self.invalidate(prefix)

    async def clear_all(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of keys deleted
        """
        return await self.invalidate("*")

    async def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats (available, entry count)
        """
        if not self.redis_client:
            return {"available": False}

        try:
            cursor = 0
            count = 0
            total_size = 0
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor, match="cache:*", count=100
                )
                count += len(keys)
                if cursor == 0:
                    break
            return {
                "available": True,
                "cached_entries": count,
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key (without 'cache:' prefix)

        Returns:
            Cached value or None
        """
        if not self.redis_client:
            return None

        try:
            full_key = f"cache:{key}" if not key.startswith("cache:") else key
            value = await self.redis_client.get(full_key)
            if value:
                return json.loads(value)
        except Exception:
            pass
        return None

    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        if not self.redis_client:
            return False

        try:
            full_key = f"cache:{key}" if not key.startswith("cache:") else key
            await self.redis_client.setex(full_key, ttl, json.dumps(value, default=str))
            return True
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete a specific cache entry.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        if not self.redis_client:
            return False

        try:
            full_key = f"cache:{key}" if not key.startswith("cache:") else key
            await self.redis_client.delete(full_key)
            return True
        except Exception:
            return False
