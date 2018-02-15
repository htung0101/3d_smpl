from smpl_webuser.serialization import load_model
import numpy as np
import inspect
import math
## Assign random pose and shape parameter
from numpy.linalg import norm
import math

def avg_joint_error(m):
  return np.mean(np.sqrt(np.sum(np.square(m),0)))

def get_valid_rotation(R):
  if isRotationMatrix(R):
    return R, 0
  U, s, V  = np.linalg.svd(R)
  new_R = np.matmul(U, V)
  if not abs(np.linalg.det(new_R) - 1) < 0.01:
    U[end, :] = (-1) * U[end, :]
    new_R = np.matmul(U, V)

  return new_R, np.linalg.norm(new_R - R) 

def isRotationMatrix(R):
  Rt = np.transpose(R)
  shouldBeIdentity = np.dot(Rt, R)
  I = np.identity(3, dtype = R.dtype)
  n = np.linalg.norm(I - shouldBeIdentity)
  return n < 1e-6

def rotationMatrixToEulerAngles(R, print_error=False) :
  R, recovered_error = get_valid_rotation(R)
  assert(isRotationMatrix(R))
  
  sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
  singular = sy < 1e-6

  if  not singular :
    x = math.atan2(R[2,1] , R[2,2])
    y = math.atan2(-R[2,0], sy)
    z = math.atan2(R[1,0], R[0,0])
  else:
    x = math.atan2(-R[1,2], R[1,1])
    y = math.atan2(-R[2,0], sy)
    z = 0
  if print_error:
    return np.array([x, y, z]), rocovered_error
  else:
    return np.array([x, y, z])



def eulerAnglesToRotationMatrix(theta) :
  R_x = np.array([[1,         0,                  0                   ],
                  [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                  [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                   ])
  R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                  [0,                     1,      0                   ],
                  [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                  ])
  R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
    [math.sin(theta[2]),    math.cos(theta[2]),     0],
    [0,                     0,                      1]
    ])
  R = np.dot(R_z, np.dot( R_y, R_x ))
  return R

