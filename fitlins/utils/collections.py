def dict_intersection(dict1, dict2):
    return {k: v for k, v in dict1.items() if dict2.get(k) == v}
