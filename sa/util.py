from math import sqrt

def distance(object1, object2):
    diff_x = object1.x - object2.x
    diff_y = object1.y - object2.y
    return sqrt(diff_x * diff_x + diff_y * diff_y)

