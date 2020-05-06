Python 3.7.7 (tags/v3.7.7:d7c567b08f, Mar 10 2020, 10:41:24) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> x = ['apple', 'Banana', 'tomato']
>>> z = [k for j,k in enumerate(x)]
>>> z
['apple', 'Banana', 'tomato']
>>> z = [j for j,k in enumerate(x)]
>>> z
[0, 1, 2]
>>> c = [1.0, 0.0, 0.0]
>>> z = [j for j,k in enumerate(c)]]
SyntaxError: invalid syntax
>>> z = [j for j,k in enumerate(c)]
>>> z
[0, 1, 2]
>>> z = [k for j,k in enumerate(c)]
>>> z
[1.0, 0.0, 0.0]
>>> type(z)
<class 'list'>
>>> import numpy as np
>>> 0.5*1*np.cos(2*pi*0*1/(32-1))
Traceback (most recent call last):
  File "<pyshell#13>", line 1, in <module>
    0.5*1*np.cos(2*pi*0*1/(32-1))
NameError: name 'pi' is not defined
>>> 0.5*1*np.cos(2*np.pi*0*1/(32-1))
0.5
>>> 0.5*1*np.cos(2*np.pi*0*1/(32-1)) + 0.5*(-1)**1*np.cos(2*np.pi*k*n/(32-1))
Traceback (most recent call last):
  File "<pyshell#15>", line 1, in <module>
    0.5*1*np.cos(2*np.pi*0*1/(32-1)) + 0.5*(-1)**1*np.cos(2*np.pi*k*n/(32-1))
NameError: name 'k' is not defined
>>> 0.5*1*np.cos(2*np.pi*0*1/(32-1)) + 0.5*(-1)**1*np.cos(2*np.pi*(1)*1/(32-1))
0.010235029373752758
>>> 0.5*1*np.cos(2*np.pi*0*(2)/(32-1)) + 0.5*(-1)**1*np.cos(2*np.pi*1*(1)/(32-1))
0.010235029373752758
>>> 0.5*1*np.cos(2*np.pi*0*(2)/(32-1)) + 0.5*(-1)**1*np.cos(2*np.pi*1*(2)/(32-1))
0.0405210941898847
>>> 