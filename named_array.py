import numpy as np
from pylab import where

class AxisArray(np.ndarray):
    def __new__(cls, input_array, column_names=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.column_names = np.asarray(column_names)
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.column_names = getattr(obj, 'column_names', None)

    def get_axis(self, name):
        return self[:, where(self.column_names==name)[0]].flatten()
