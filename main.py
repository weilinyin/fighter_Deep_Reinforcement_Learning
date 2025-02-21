from MyEnvs import FighterEnv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


myenv = FighterEnv()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(myenv.plotdata["defender"]["x"] , myenv.plotdata["defender"]["y"],myenv.plotdata["defender"]["z"] ,label='parametric curve')
ax.plot(myenv.plotdata["fighter"]["x"] , myenv.plotdata["fighter"]["y"],myenv.plotdata["fighter"]["z"] ,label='parametric curve')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.set_box_aspect([1,1,1])




plt.figure()
plt.plot(myenv.plotdata["defender"]["x"] , myenv.plotdata["defender"]["y"])
plt.plot(myenv.plotdata["fighter"]["x"] , myenv.plotdata["fighter"]["y"])

plt.figure()
plt.plot(myenv.plotdata["defender"]["x"] , myenv.plotdata["defender"]["z"])
plt.plot(myenv.plotdata["fighter"]["x"] , myenv.plotdata["fighter"]["z"])

plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["theta"])
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["theta"])

plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["psi"])
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["psi"])

plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["a_y"])
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["a_y"])

plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["a_z"])
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["a_z"])



plt.show()