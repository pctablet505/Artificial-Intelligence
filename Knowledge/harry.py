
from logic import *

rain=Symbol('rain')
hagrid=Symbol('hagrid')
dumbledore=Symbol('dumbledore')

'''
If it doesn't rain then harry goes to hagrid.
Harry goes either to hagrid or dumbledore
Harry goes to dumbledore

'''


knowledge=And(Implication(Not(rain),hagrid),
              Or(hagrid,dumbledore),
              Not(And(hagrid,dumbledore)),
              dumbledore)
print(knowledge.formula())
print(model_check(knowledge,rain))






