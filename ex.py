import numpy as np
'''

altitude = input('Enter drone altitude: ')
sensor_w = input('Enter sensor width: ')
sensor_h = input('Enter sensor height: ')
focal_l = input('Enter focal length: ')

# Image dimension will be retrieved from YOLO result when integrated
image_w = input('Enter image width: ')
image_h = input('Enter image height: ')

gsd_h = (altitude * sensor_h) / (focal_l * image_h)
gsd_w = (altitude * sensor_w) / (focal_l * image_w)

print(f"GSD value between GSD(Height) : {gsd_h} and GSD{Width} : {gsd_w}")
'''
# Replace with mask array
test_arr = np.array([[[0, 0, 1, 1], [1, 0, 1, 1]], [[0, 0, 1, 0], [1, 0, 1, 1]]])
print(f"Value for mask is {np.count_nonzero(test_arr, axis=(1,2))}")
