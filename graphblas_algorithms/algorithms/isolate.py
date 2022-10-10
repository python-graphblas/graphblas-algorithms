__all__ = ["is_isolate", "isolates", "number_of_isolates"]


def is_isolate(G, n):
    index = G._key_to_id[n]
    if G.is_directed():
        degrees = G.get_property("total_degrees+")
    else:
        degrees = G.get_property("degrees+")
    return index not in degrees


def isolates(G):
    if G.is_directed():
        degrees = G.get_property("total_degrees+")
    else:
        degrees = G.get_property("degrees+")
    return (~degrees.S).new(name="isolates")


def number_of_isolates(G):
    if G.is_directed():
        degrees = G.get_property("total_degrees+")
    else:
        degrees = G.get_property("degrees+")
    return degrees.size - degrees.nvals
