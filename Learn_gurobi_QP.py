import gurobipy as gp
from gurobipy import GRB
import numpy as np

## QP
m=gp.Model('qp')

x=m.addVar(ub=1.0,name='x')
y=m.addVar(ub=1.0,name='y')
z=m.addVar(ub=1.0,name='z')

obj=x*x+x*y+y*y+y*z+z*z+2*x
m.setObjective(obj)

m.addConstr(x+2*y+3*z>=4,'c0')
m.addConstr(x+y>=1,'c1')

m.optimize()
print(m.getVars())

ans=[]
for v in m.getVars():
    ans.append(v.x)
ans=np.array(ans)
print(ans)

## SOS


## Constraints
