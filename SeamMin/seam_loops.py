"""
Create a seam loop from a list of edge pairs.

Written by Zachary Ferguson
"""

from util import verts_equal


def find_seam_loops(edgepairs):
    """
    Creates a seam loop from a edgePair object.
    Parameters: edgepairs - tuple of edgepairs
    Returns: List of list containing seam loops
    """
    # Flatten the edgepairs
    bag_of_edges = [edge for edgepair in edgepairs for edge in edgepair]
    return find_edge_loops(bag_of_edges)


def find_edge_loops(bag_of_edges):
    """ Find the edge loops given an unsorted 1D list of edges """
    if(len(bag_of_edges) == 0):
        return []

    # Create edge loops from bag_of_edges
    currentEdge = bag_of_edges.pop(0)
    currentLoop = 0
    edge_loops = [[currentEdge]]
    while len(bag_of_edges) > 0:
        nextEdge = None
        # Loop over the remaining edges and find the next edge
        for i, edge in enumerate(bag_of_edges):
            if(currentEdge[1] == edge[0]):
                nextEdge = bag_of_edges.pop(i)
                break

        # Store the found edge in the current seam loop
        if(nextEdge):
            edge_loops[currentLoop].append(nextEdge)

        # Initialize the next seam loop if needed
        if len(bag_of_edges) > 0 and ((nextEdge is None) or
                nextEdge[1] == edge_loops[currentLoop][0][0]):
            currentEdge = bag_of_edges.pop(0)
            edge_loops.append([currentEdge])
            currentLoop += 1
        # Increment the currentEdge
        elif(len(bag_of_edges) > 0):
            currentEdge = nextEdge

    return edge_loops
