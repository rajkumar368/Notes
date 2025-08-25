# NumPy All-in-One Notes  

```python
# =========================================================
# 0. IMPORT
# =========================================================
import numpy as np

# =========================================================
# 1. CREATING ARRAYS
# =========================================================
a = np.array([1, 2, 3])                       # 1-D
b = np.array([[1, 2], [3, 4]])                # 2-D
c = np.array([[[1, 2]], [[3, 4]]])            # 3-D

# quick helpers
np.zeros((2, 3))
np.ones((2, 3))
np.empty((2, 3))
np.full((2, 3), 7)
np.eye(3)                                     # identity
np.arange(0, 10, 2)                           # like range
np.linspace(0, 1, 5)                          # 0..1 in 5 steps
np.random.rand(2, 3)                          # uniform [0,1)
np.random.randn(2, 3)                         # normal
np.random.randint(0, 10, (2, 3))
np.fromfunction(lambda i, j: i + j, (3, 3))
np.fromiter(range(5), dtype=int)

# =========================================================
# 2. INSPECTING
# =========================================================
a.shape
a.ndim
a.size
a.dtype
a.itemsize
a.nbytes
a.T                                           # transpose
np.info(np.ndarray)                           # docstring

# =========================================================
# 3. INDEXING & SLICING
# =========================================================
a = np.arange(10)
a[0], a[-1], a[2:5], a[::-1]
b = np.arange(12).reshape(3, 4)
b[1, 2], b[1, :], b[:, 2], b[1:3, 0:2]

# boolean indexing
mask = b % 2 == 0
b[mask]

# fancy indexing
b[[0, 2], [1, 3]]

# =========================================================
# 4. MUTATING
# =========================================================
a = np.arange(5)
a[0] = 99
a[1:3] = [88, 77]
np.append(a, [9, 8])
np.insert(a, 1, 100)
np.delete(a, 0)
np.unique(a)

# =========================================================
# 5. SHAPE MANIPULATION
# =========================================================
a = np.arange(6)
b = a.reshape(2, 3)
b.ravel()                                     # flatten
b.flatten()
b.transpose()
b.swapaxes(0, 1)
np.expand_dims(b, axis=0)                     # (1,2,3)
np.squeeze(np.zeros((1,2,1)))                 # remove 1-D axes

# =========================================================
# 6. BROADCASTING
# =========================================================
a = np.array([1, 2, 3])
b = np.array([[10], [20]])
a + b                                           # (2×3)

# =========================================================
# 7. ELEMENT-WISE MATH
# =========================================================
a = np.arange(1, 5)
np.add(a, a)
np.subtract(a, a)
np.multiply(a, a)
np.divide(a, 2)
np.power(a, 2)
np.sqrt(a)
np.exp(a)
np.log(a)
np.sin(a)
np.cos(a)
np.tan(a)
np.floor(a)
np.ceil(a)
np.round(a, 2)

# =========================================================
# 8. AGGREGATIONS
# =========================================================
a = np.arange(1, 10)
a.sum()
a.mean()
a.std()
a.var()
a.min()
a.max()
a.argmin()
a.argmax()
a.cumsum()
a.cumprod()
np.percentile(a, 75)
np.quantile(a, 0.75)
np.median(a)

# axis wise
b = np.arange(12).reshape(3, 4)
b.sum(axis=0)     # columns
b.sum(axis=1)     # rows

# =========================================================
# 9. SORTING & SEARCHING
# =========================================================
a = np.array([3, 1, 2])
np.sort(a)
a.sort()
np.argsort(a)
np.lexsort([a, -a])
np.searchsorted([1, 2, 3, 4], 2.5)

# =========================================================
# 10. LINEAR ALGEBRA
# =========================================================
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])
np.dot(A, B)            # or A @ B
np.matmul(A, B)
np.vdot(A, B)
np.inner(A, B)
np.outer(A[:, 0], B[:, 0])
np.tensordot(A, B, axes=1)
np.linalg.det(A)
np.linalg.inv(A)
np.linalg.eigvals(A)
np.linalg.solve(A, [1, 2])
np.linalg.svd(A)

# =========================================================
# 11. RANDOM SAMPLING
# =========================================================
np.random.seed(42)
np.random.rand(2, 3)
np.random.randn(2, 3)
np.random.randint(0, 10, (2, 3))
np.random.choice([10, 20, 30], size=5, replace=True, p=[.5, .3, .2])

# =========================================================
# 12. BOOLEAN / MASKING
# =========================================================
a = np.arange(10)
mask = (a > 3) & (a < 8)
a[mask]
np.where(a > 5, 1, 0)
np.any(mask)
np.all(mask)

# =========================================================
# 13. SET OPERATIONS
# =========================================================
a = np.array([1, 2, 3, 2])
b = np.array([2, 3, 4])
np.unique(a)
np.intersect1d(a, b)
np.setdiff1d(a, b)
np.union1d(a, b)

# =========================================================
# 14. CLIPPING / ROUNDING / DIGITS
# =========================================================
a = np.array([1, 2, 3, 4, 5])
np.clip(a, 2, 4)
np.round(np.random.rand(3) * 10, 2)

# =========================================================
# 15. SAVE / LOAD ARRAYS
# =========================================================
np.save('arr.npy', a)
a = np.load('arr.npy')
np.savetxt('arr.txt', a)
a = np.loadtxt('arr.txt')

# =========================================================
# 16. MEMORY VIEWS VS COPIES
# =========================================================
a = np.arange(5)
b = a            # view
c = a.copy()     # deep copy
b[0] = 999       # changes a

# =========================================================
# 17. DATETIME64 & TIMEDELTA64
# =========================================================
dates = np.arange('2025-01-01', '2025-01-10', dtype='datetime64[D]')
deltas = np.array([1, 2, 3], dtype='timedelta64[D]')
dates + deltas

# =========================================================
# 18. STRUCTURED ARRAYS
# =========================================================
dt = np.dtype([('name', 'U10'), ('age', 'i4')])
people = np.array([('Alice', 25), ('Bob', 30)], dtype=dt)
people['age']
people[people['age'] > 27]

# =========================================================
# 19. PERFORMANCE TIPS
# =========================================================
# vectorized vs loops
a = np.arange(1e6)
%timeit [i*2 for i in a]          # slow
%timeit a * 2                     # fast

# =========================================================
# 20. QUICK ONE-LINERS
# =========================================================
# create grid
x, y = np.meshgrid(np.arange(3), np.arange(2))
# one-hot
np.eye(3)[np.arange(3)]
# repeat
np.repeat([1, 2, 3], 2)
# tile
np.tile([1, 2], 3)
# diag
np.diag([1, 2, 3])
# flatten
np.ravel_multi_index([[0, 1], [2, 3]], (4, 4))
# histogram
hist, bins = np.histogram(np.random.randn(1000), bins=10)
