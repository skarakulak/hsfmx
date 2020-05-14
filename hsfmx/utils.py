def get_most_freq_indices(list_of_indices,index_map_dict):
    new_indices = []
    for index in list_of_indices:
        try:
            new_indices.append(index_map_dict[index])
        except KeyError:
            print("This shouldn't have happened. An index was not found in the index map. Currently changed to unknown tag but needs to be checked.")
            new_indices.append(0)

    assert len(new_indices)==len(list_of_indices)
    return new_indices