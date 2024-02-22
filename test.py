import numpy as np

original = np.array([1, 2, 3, 4, 5, 5])
# Example array with NaN values
arr = np.array([1, np.nan, 3, np.nan, 5, np.nan])

# Create a mask where True indicates non-NaN values
mask = ~np.isnan(arr)
result = np.where(mask, True, np.nan)
test_output = original * result

# Replace NaN values with False and non-NaN values with True


print(mask)
print(result)
print(test_output)