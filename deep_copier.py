test_x = [1,2,3,[4,5],6,[7,8,9,[10,11]]]

def deep_copier(x_list):
    rep_x_list = []
    for x in x_list:
        try:
            iterator_x = iter(x)
            rep_x_list.append(deep_copier(x))
        except TypeError:
            rep_x_list.append(x)
    return rep_x_list

copied_test_x = deep_copier(test_x)

copied_test_x[3][1]=0
print(test_x)
print(copied_test_x)