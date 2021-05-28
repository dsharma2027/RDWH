#!/usr/bin/env python
# coding: utf-8

# # Linear Algebra

# In[1]:


# Importing the Libraries required 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d
import sympy
import scipy
import scipy.linalg
import math


# <font color=red> **1. Basic Matrix Manipulation** </font>

# <font color = blue> **a) Matrix Addition**

# In[2]:


A1 = np.mat('-1,4,-6;8,5,16;2,8,5')
B1 = np.mat('12,7,6;8,0,5;3,2,4')
C1 = np.mat('2,17,-2;18,0,-4;0,12,-4')


# In[3]:


# Matrix A
sympy.Matrix(A1)


# In[4]:


# Matrix B
sympy.Matrix(B1)


# In[5]:


# Matrix C
sympy.Matrix(C1)


# In[6]:


# Addition Of Two Matrix

A1 + B1  # Addition can be done with simple '+' operator between 2 matrices


# In[7]:


B1+A1


# In[8]:


# Matrix Addition follows Associative Properties

(A1+B1)+C1 == A1+(B1+C1)


# <font color = blue> **b) Matrix Subtraction**

# In[9]:


# Subtraction Of Two Matrix

A1 - B1  # Subtraction can be done with simple '-' operator between 2 matrices


# In[10]:


# Matrix Substraction does not follows Associative Properties

(A1-B1)-C1 == A1-(B1-C1)


# In[ ]:





# <font color = blue> **c) Matrix Multiplication**

# In[11]:


np.matmul(A1,B1) # We can use the numpy method matmul


# In[12]:


A1@B1    # Or we can use the '@' operator


# In[13]:


np.dot(A1,B1)  # We can also use dot product


# In[14]:


# AB != BA

(A1@B1) == (B1@A1)


# In[15]:


# Matrix Multiplication follows Associative Properties

(A1@B1)@C1 == A1@(B1@C1)


# <font color = blue> **d) Determinant Of Matrix** </font>
#     <br>
#   > - Used for Matrix Inversion <br> 
#   > - If det(A) = 0, then A has no Inverse 

# In[16]:


# Determinant of 3X3 Matirx by Row Expansion Method
# we expanded using the first row. This can be done using any row or any column

# Determinent when row 1 is expanded

minor_11 = (np.round((np.linalg.det(A1[np.ix_([1,2],[1,2])])),0))
minor_12 = (np.round((np.linalg.det(A1[np.ix_([1,2],[0,2])])),0))
minor_13 = (np.round((np.linalg.det(A1[np.ix_([1,2],[0,1])])),0))

det_row1 = ((A1[0,0]*(minor_11)) + ((-1)*(A1[0,1])*(minor_12)) + (A1[0,2])*(minor_13))
det_row1


# In[17]:


# Determinant Using Numpy Method Det
np.round(np.linalg.det(A1))


# In[ ]:





# <font color = blue> **e) Transpose Of Matrix** </font>
# > The transpose of a matrix is an operator which flips a matrix over its diagonal; that is, it switches the row and column indices of the matrix A by producing another matrix

# In[18]:


# Transpose of Matrix A1 using numpy method

np.transpose(A1)


# In[19]:


A1


# In[20]:


# Is Det(A transpose) = Det(A)?

np.linalg.det(np.transpose(A1)) == np.linalg.det(A1)


# In[21]:


# Transpose(A+B) = transpose(A) +Transpos(B)  (Symmetric)

(A1+B1).T == A1.T + B1.T


# In[22]:


# Transpose(A-B) = transpose(A) -Transpos(B)   (Skew Symeetric)
(A1-B1).T == A1.T - B1.T


# <font color = blue> **e) Inverse Of Matrix** </font>

# In[23]:


# Inverse of Matrix using Scipy Method

scipy.linalg.inv(A1)


# In[24]:


# Using Numpy Method
A1.I


# In[25]:


# Matrix is Invertible if Its determinant is not zer0

np.linalg.det(A1)


# In[26]:


# Inv(Inv(A)) = A

np.round((A1.I).I) == A1


# <font color = blue> **f) Adjoint Of Matrix** </font>
# > inv(A) = Adj(A)/Det(A) <br>
# > Adj(A) = Inv(A).Det(A)

# In[27]:


# Adjoint of A

A1_Adj = (A1.I) * (np.linalg.det(A1))
A1_Adj


# In[28]:


# Multiplication of A1 and Adj(A1) is a diagnol matrix with det(A1) as its diagnol elements
np.round(A1@A1_Adj)


# =====================================================================================================================

# .

# <font color=red> **2. Vectors** </font>

# <font color = blue> **a) Re-Presenting Vectors In 2-D Space** </font>

# In[29]:


vecs = ((2, 4), (-3, 3), (-4, -3.5))   # In Pyhton We can also pass Vectors as Tuple

fig, ax = plt.subplots(figsize=(10, 8))
# Set the axes through the origin
for spine in ['left', 'bottom']:
    ax.spines[spine].set_position('zero')
for spine in ['right', 'top']:
    ax.spines[spine].set_color('none')

ax.set(xlim=(-5, 5), ylim=(-5, 5))
ax.grid()
for v in vecs:
    ax.annotate('', xy=v, xytext=(0, 0),
                arrowprops=dict(facecolor='blue',
                shrink=0,
                alpha=0.7,
                width=0.5))
    ax.text(1.1 * v[0], 1.1 * v[1], str(v))
plt.show()


# In[ ]:





# <font color = blue> **b) Magnitude of the Vector** </font>

# In[30]:


V1 = np.array([5,7])
V1


# In[31]:


# Magnitude of the Vector

V1_Mag = ((V1**2).sum())**0.5
print(f"The Magnitude of the Vector {V1} is {V1_Mag}")


# In[32]:


# Converting In into the Unit Vector

V1_hat = V1/V1_Mag
print(f"The Unit vector of {V1} is {V1_hat}")


# In[33]:


# Magnitude of the Unit Vector is 1

V1_hat_Mag = ((V1_hat**2).sum())**0.5
V1_hat_Mag


# In[ ]:





# <font color = blue> **b) Norm of the Vector** </font>
# > - L1 norm (Manhattan Distance) <br>
# > - L2 norm (Eucilidean Distance)

# In[34]:


V2 = np.array([5,6,9])
V2


# In[35]:


# L1 Norm
V2_L1norm = np.linalg.norm(V2,1)

# L2 Norm
V2_L2norm = np.linalg.norm(V2)

print(f"The L1 norm of {V2} is {V2_L1norm}")
print(f"The L2 norm of {V2} is {V2_L2norm}")


# In[ ]:





# In[103]:


import pylab


def plotUnitCircle(p):
 """ plot some 2D vectors with p-norm < 1 """
 for i in range(5000):
  x = np.array([np.random.rand()*2-1,np.random.rand()*2-1])
  if np.linalg.norm(x,p) < 1:
   pylab.plot(x[0],x[1],'bo')
 pylab.axis([-1.5, 1.5, -1.5, 1.5])
 pylab.show()


# In[104]:


# 1 Norm
plotUnitCircle(1)


# In[105]:


# 2 norm
plotUnitCircle(2)


# <font color = blue> **c) Multiplication of Vector by Scalar** </font>

# <b> Find the Vector in the dfirection of vector [5,-1,2] which has magnitude of 8 units.

# In[40]:


A1 = np.array([5,-1,2])
s = 8
A1_hat = A1/(np.linalg.norm(A1))
print(f"The vector in the direction of {A1} with magnitude {s} is {s*A1_hat}" )


# -------------------------------------------------------------------------------------------------------------------

# 

# <font color = blue> **d) Dot Product** </font>

# In[41]:


A2 = np.array([1,-2,3])
B2 = np.array([3,-2,1])


# In[42]:


# Calculating the Dot product between A2 and B2
dotP2 = np.dot(A2,B2)


# In[43]:


# Calculating the magnitude of A2 and B2

A2_mag = ((A2**2).sum())**0.5
B2_mag = ((B2**2).sum())**0.5


# In[44]:


# Calculating the angle between the vectors
# A2.B2 = |A2|*|B2|*Cos(theta)
# Cos(theta) = (A2.B2)/|A2|*|B2|
# theta = Cos Inv((A2.B2)/|A2|*|B2|)

costheta = dotP2 / (A2_mag * B2_mag)
theta = np.degrees(np.arccos(costheta))

print(f"The dot product of vector {A2} and {B2} is {dotP2}")
print(f"The magnitude of {A2} is {np.round(A2_mag,2)}")
print(f"The magnitude of {B2} is {np.round(B2_mag,2)}")
print(f"The Angle between vector {A2} and {B2} is {np.round(theta,2)}")



# In[ ]:





# <font color = blue> **e) Span Of Vector** </font>
# 
# > - Given a set of vectors A:={a1,…,ak} in Rn, it’s natural to think about the new vectors we can create by performing linear operations.
# New vectors created in this manner are called linear combinations of A.
# > - In particular, y∈Rn is a linear combination of A:={a1,…,ak} if
# y=β1a1+⋯+βkak for some scalars β1,…,βk
# In this context, the values β1,…,βk are called the coefficients of the linear combination.
# The set of linear combinations of A is called the span of A.

# In[107]:


fig = plt.figure(figsize=(10, 8))
ax = fig.gca(projection='3d')

x_min, x_max = -5, 5
y_min, y_max = -5, 5

a, b = 0.2, 0.1

ax.set(xlim=(x_min, x_max), ylim=(x_min, x_max), zlim=(x_min, x_max),
       xticks=(0,), yticks=(0,), zticks=(0,))

gs = 3
z = np.linspace(x_min, x_max, gs)  # Return Evenly Spaced nos.
x = np.zeros(gs)
y = np.zeros(gs)
ax.plot(x, y, z, 'k-', lw=2, alpha=0.5)
ax.plot(z, x, y, 'k-', lw=2, alpha=0.5)
ax.plot(y, z, x, 'k-', lw=2, alpha=0.5)


# Fixed linear function, to generate a plane
def linear_func(x, y):
    return a * x + b * y

# Vector locations, by coordinate
x_coords = np.array((3, 3))
y_coords = np.array((4, -4))
z = linear_func(x_coords, y_coords)
# adding text to the axis at the location x_cords,y_cords
for i in (0, 1):
    ax.text(x_coords[i], y_coords[i], z[i], f'$a_{i+1}$', fontsize=14)  

# Lines to vectors
for i in (0, 1):
    x = (0, x_coords[i])
    y = (0, y_coords[i])
    z = (0, linear_func(x_coords[i], y_coords[i]))
    ax.plot(x, y, z, 'b-', lw=1.5, alpha=0.6)


# Draw the plane
grid_size = 20
xr2 = np.linspace(x_min, x_max, grid_size)
yr2 = np.linspace(y_min, y_max, grid_size)
x2, y2 = np.meshgrid(xr2, yr2)
z2 = linear_func(x2, y2)
ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=cm.jet,
                linewidth=0, antialiased=True, alpha=0.2)
plt.show()


# In[ ]:





# <font color=red> **3. Solving Linear Equation** </font>

# <font color = blue> **a) Graphical Method** </font>

# We will consider below two equations:<br>
# 0.5x - y = 1<br>
# x + y = 4

# In[46]:


# Matris A will be

matA = np.mat('0.5,-1;1,1')
matA


# The Column Figure Give us<br>
# x[0.5 ,1] + y[-1,1] = [1,-4] <br>
# 
# 
# The goal is to find the value of x and y for which the linear combination of the vector [0.,5,1] and [-1,1] gives the vector [-1,4]<br>
# <b> We will sole the system graphically by plotting the equation and looking for their intersection

# In[47]:



x = np.arange(-10, 10)
y = 0.5*x + 1

y1 = -x + 4

plt.figure()
plt.grid()
plt.plot(x, y)
plt.plot(x, y1)
plt.xlim(-2, 10)
plt.ylim(-2, 10)
# draw axes
plt.axvline(x=0, color='#A9A9A9')
plt.axhline(y=0, color='#A9A9A9')
plt.show()
plt.close()


# We can see that the solution (the intersection of the lines representing our two equations) is x=2 and y=2. This means that the linear combination is the following:<br>
# 2[0.5,1] + 2[−1,1] = [−1,4]

# In[ ]:





# In[ ]:





# <font color = blue> **b) Determinant Method** </font>

# <b>
#     x + 3y - 2z = 5 <br>
#     3x + 5y + 6z = 7 <br>
#     2x + 4y + 3z = 8

# In[49]:


# Coefficent Matrix
A3 = np.mat('1,3,-2;3,5,6;2,4,3')
A3


# In[50]:


# COnstant Matrix B, containing elements from right hand side of matrix

B3 = np.mat('5;7;8')
B3


# In[51]:


# Checking Determinant of Matrix A

np.linalg.det(A3)


# <b> The determinant of above system of Equations is not equal to zero,<br> Therfor the system is consistent and has Unique Solution.

# In[52]:


# Calculation Dx
A3_x = A3.copy()
A3_x[:,0] = B3
Dx = np.round(np.linalg.det(A3_x))


# In[53]:


# Calculation Dy
A3_y = A3.copy()
A3_y[:,1] = B3
Dy = np.round(np.linalg.det(A3_y))


# In[54]:


# Calculation Dz
A3_z = A3.copy()
A3_z[:,2] = B3
Dz = np.round(np.linalg.det(A3_z))


# In[55]:


print("The Value of x is: ", Dx)
print("The Value of y is: ", Dy)
print("The Value of z is: ", Dz)


# In[ ]:





# <font color = blue> **c) Matrix Inverse Method** </font>

# <b>
#     x + 2y + 5z = 10<br>
#     2x + 5y + z = 8<br>
#     2x + 3y + 8z = 5

# In[56]:


# Coeeficent Matrix

A4 = np.mat('1,2,5;2,5,1;2,3,8')
A4


# In[57]:


# Matrix B

B4 = np.mat('10;8;5')
B4


# <b> AX = B<br>
#     X = Inv(A)B

# In[58]:


# Inverse Of A
A4_inv = A4.I
A4_inv


# In[59]:


X4 = A4_inv @ B4
print("The solution of x is: ", np.round(X4[0,0],2))
print("The solution of y is: ", np.round(X4[1,0],2))
print("The solution of z is: ", np.round(X4[2,0],2))


# In[ ]:





# <font color = blue> **d) RREF Method** </font>

# <b>
#     x + y - z =4<br>
#     x - 2y + 3z = -6<br>
#     2x + 3y +z = 7

# In[60]:


# Coefficent Matrix

A5 = np.mat('1,1,-1;1,-2,3;2,3,1')
A5


# In[61]:


# Matrix B

B5 = np.mat('4;-6;7')
B5


# In[62]:


Aug_Mat = sympy.Matrix(np.hstack([A5,B5]))
Aug_Mat


# In[63]:


rref,_ = Aug_Mat.rref()
rref


# In[ ]:





# <font color=red> **3. Linear Independency** </font>
# 
# > The following statements are equivalent to linear independence of A:={a1,…,ak}⊂Rn
#     > - No vector in A can be formed as a linear combination of the other elements.
#     > - If β1a1+⋯βkak=0 for scalars β1,…,βk, then β1=⋯=βk=0. <br>
#     (The zero in the first expression is the origin of Rn)

# <b> V1 = [1,-2,0]<br> V2 = [4,0,8]<br> V3 = [3,-1,5]

# In[64]:


# MAtris A

A6 = np.mat('1,4,3;-2,0,-1;0,8,5')
A6


# In[65]:


#  Zero Matrix, For Vectrs to be independent, a1v1 +a1v2 +  a3v3  = 0
B6 = np.mat('0;0;0')


# In[66]:


Aug6 = sympy.Matrix(np.hstack([A6,B6]))
Aug6


# In[67]:


rref6 = Aug6.rref()
rref6


# <b> As we can see from above rref we can se that  for  the above 3 vectors we got Non trivial solution<br>
#     There for the three vectors are Linearly Dependent

# In[ ]:





# <b> 
#     U1 = [1,2,1,4]<br>
#     U2 = [0,1,1,1]<br>
#     U3 = [2,0,1,7]

# In[68]:


#Matrix A

A7 = np.mat('1,0,2;2,1,0;1,1,1;4,1,7')
A7


# In[69]:


B7 = np.mat('0;0;0;0')


# In[70]:


aug7 = sympy.Matrix(np.hstack([A7,B7]))
aug7


# In[71]:


rref7 = aug7.rref()
rref7


# <b> As we can see from above rref we can se that  for  the above 3 vectors we got trivial solution<br>
#     There for the three vectors are Linearly Independent

# In[ ]:





# <font color=red> **4. Linear Transformation and Basis Of the Vector Space** </font>
# 
# > The basis of a vectors space is a set of vectors which are:
#   > - independent vectors
#   > - span the vector space
# 

# <font color = blue> **a) Change Of Basis** </font>

# <b> a vector [(2,7),(1,5)] has to be projected from unit basis (usual i,j) to a new basis. The new basis is made up of two vectors     [2,1] and [-1,3]

# In[72]:


V3 = np.array([[2,1],[7,5]])
V3


# In[73]:


old_basis = np.array([[1,0],[0,1]])
old_basis


# In[74]:


new_basis = np.array([[2,-1],[1,3]])
new_basis


# <b>new_basis . V3' = old_basis . V3<br>
#    V3' = Inv(neW_basis) . old_basis . V3 

# In[75]:


# V3 In new basis is

V3_new = np.round((np.linalg.inv(new_basis))@(old_basis @ V3),2)
V3_new


# <font color = blue> **b) To find the Basis of the set of Vectors** </font>

# <b> Find the basis for the below set of Vectors<br>
#     [1,2,1,2],[-1,-1,1,-1],[1,4,5,3],[3,4,-1,6],[0,1,2,5],[4,4,6,3]

# In[79]:


matB = np.mat('1,-1,1,3,0,4;2,-1,4,4,1,4;1,1,5,-1,2,6;2,-1,3,6,5,3')
matB


# The Above vectors are not linearly Independent as no. of rows are less than no. of vectors

# In[80]:


mat0 = np.mat('0;0;0;0')
mat0


# In[81]:


augB = sympy.Matrix(np.hstack([matB,mat0]))
augB


# In[82]:


rrefB = augB.rref()
rrefB


# <b> Vectors 1,2,3 adnd 6  ([1,2,1,2],[-1,-1,1,-1],[3,4,-1,6],[4,4,6,3]) provide the basis of the set vecrtors

# In[ ]:





# ======================================================================================================================

# ## Linear Algebra Exercise

# <font color = green> **1) Write a NumPy program to compute the eigenvalues and right eigenvectors of a given square array..** </font><br>
# 

# In[83]:


matA = np.mat("4,5;3,6")
print("Original matrix:")
print("a\n", matA)
w, v = np.linalg.eig(matA) 
print( "Eigenvalues of the said matrix",w)
print( "Eigenvectors of the said matrix",v)


# ------------------------------------------------------------------------------------------------------------------

# <font color = green> **2)Write a NumPy program to compute the condition number of a given matrix.** </font><br>
# > - In the field of numerical analysis, the condition number of a function with respect to an argument measures how much the output value of the function can change for a small change in the input argument. This is used to measure how sensitive a function is to changes or errors in the input, and how much error in the output results from an error in the input.
# > - In linear regression the condition number can be used as a diagnostic for multicollinearity.
# > -  In non-mathematical terms, an ill-conditioned problem is one where, for a small change in the inputs (the independent variables or the right-hand-side of an equation) there is a large change in the answer or dependent variable. This means that the correct solution/answer to the equation becomes hard to find. The condition number is a property of the problem. Paired with the problem are any number of algorithms that can be used to solve the problem, that is, to calculate the solution. 

# In[84]:


a = np.array([[1,4,3], [-2,0,-1], [0,8,5]])
print("Original matrix:")
print(a)
print("The condition number of the said matrix:")
print(np.round(np.linalg.cond(a)))


# In[ ]:





# --------------------------------------------------------------------------------------------------------------------

# <font color = green> **3)Write a NumPy program to calculate the QR decomposition of a given matrix.** </font><br>
# > Any real square matrix A may be decomposed as
#  A = QR, \, 
# where Q is an orthogonal matrix (its columns are orthogonal unit vectors meaning Q.Qtranspose = Qtranspose . Q = I and R is an upper triangular matrix (also called right triangular matrix, hence the name). If A is invertible, then the factorization is unique if we require the diagonal elements of R to be positive.

# In[85]:


m = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("Original matrix:")
print(m)
result =  np.round(np.linalg.qr(m))
print("Decomposition of the said matrix:")
print(result)


# <font color = blue> <b>QR Decompostion is also used to find the basis of the vectors

# <b> - If A has n linearly independent columns, then the first n columns of Q form an orthonormal basis for the column space of A. More generally, the first k columns of Q form an orthonormal basis for the span of the first k columns of A for any 1 ≤ k ≤ n.[1] The fact that any column k of A only depends on the first k columns of Q is responsible for the triangular form of R.[1]

# In[86]:


A7 = np.mat('1,0,2;2,1,0;1,1,1')
A7


# In[87]:


result = np.round(np.linalg.qr(A7))
result


# From the above Q Matrix we can say thay [1,2,1] is the orthogonal basis of set of vectors.

# ---------------------------------------------------------------------------------------------------------------------

# <font color = green> **4)Write a NumPy program to get the LU Decomposition of a given array..** </font><br>
# 

# In[110]:


a = np.array([[4, 12, -16], [12, 37, -53], [-16, -53, 98]], dtype=np.int32)
print("Original array:")
print(a)
p,l,u = scipy.linalg.lu(a)
print("Lower-trianglular L in the LU decomposition of the said array:")
print(np.round(l,2))
print("Upper-trianglular L in the LU decomposition of the said array:")
print(np.round(u,2))


# In[113]:


p@l@u


# In[ ]:





# <font color = green> **5) Write a NumPy program to get the SVD of a given array..** </font><br>

# In[134]:


matA5 = np.array([[1,2],[3,4]])
matA5


# In[135]:


# Caclulating SVD

U, s, VT = scipy.linalg.svd(matA5)
print(U)
print('\n')
print(s)
print('\n')
print(VT)


# In[138]:


# Reconstruction Matrix from SVD

sigma = np.diag(s)  # Creating a nxn sigma matrix
matB5 = np.round((U.dot(sigma.dot(VT))))
print(matB5)
matA5 == matB5
        


# In[ ]:





# In[ ]:




