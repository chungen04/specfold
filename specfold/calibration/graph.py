from typing import Optional, Set, Tuple

class Node:
    """Represents a node in the draft tree."""
    def __init__(self, L: int, ij_set: Set[Tuple[int, int]], frequency: int, count: int, parent: Optional['Node'] = None):
        self.L = L
        self.ij_set = frozenset(ij_set)  # Set of (i, j) pairs representing unmasked tokens
        self.frequency = frequency
        self.count = count
        self.parent = parent
        self.children = []
        
        # Compute ancestor count sum
        self.ancestor_count_sum = 0
        if parent:
            self.ancestor_count_sum = parent.count + parent.ancestor_count_sum
    
    def get_score(self) -> int:
        """First try degree-0 method. (figure out how to do degree-1 later)"""
        return self.count
    
    def get_unmasked_count(self) -> int:
        """Number of unmasked tokens (size of ij_set)."""
        return len(self.ij_set)
    
    def __repr__(self):
        return f"Node(L={self.L}, tokens={self.ij_set}, count={self.count}, score={self.get_score()})"


def can_be_child(parent: Node, child_L: int, child_set: Set[Tuple[int, int]]) -> bool:
    """
    Check if a node can be a child of parent.
    Requirements:
    - L differs by 1: child_L = parent.L + 1
    - Child has S more unmasked tokens than parent (S=1 in base case)
    """
    if child_L != parent.L + 1:
        return False
    
    child_frozen = frozenset(child_set)
    
    # Check if child is a superset of parent (child has all parent's tokens plus more)
    if not child_frozen.issuperset(parent.ij_set):
        return False
    
    # Check if child has exactly S more tokens (S=1 in base case)
    S = 1  # Can be parameterized if needed
    if len(child_frozen) != len(parent.ij_set) + S:
        return False
    
    return True

def count_tree_nodes(node: Node) -> int:
    """Count total nodes in tree rooted at node."""
    count = 1
    for child in node.children:
        count += count_tree_nodes(child)
    return count