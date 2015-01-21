"""
Simulate a max heap by using python's built in min heap (heapq) and storing
inverted values.
"""

import heapq

def heapify(iterable):
    """Create a "max heap" by inverting every element and making a min heap."""
    for i, elem in enumerate(iterable):
        iterable[i] = ~elem+1
    heapq.heapify(iterable)

def heappop(heap):
    """Simulate popping from max heap by popping from min heap and inverting"""
    popped = heapq.heappop(heap)
    return ~popped+1

def heappush(heap, item):
    """Simulate pushing onto max heap by inverting then pushing onto min heap"""
    to_push = ~item+1
    heapq.heappush(heap, to_push)