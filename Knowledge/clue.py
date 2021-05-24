import termcolor
from logic import *

mustard = Symbol('ColMustard')
pulm = Symbol('ProfPulm')
scarlet = Symbol('MsScarlet')
characters = [mustard, pulm, scarlet]

ballroom = Symbol('ballroom')
kitchen = Symbol('kitchen')
library = Symbol('library')
rooms = [ballroom, kitchen, library]

knife = Symbol('knife')
revolver = Symbol('revolver')
wrench = Symbol('wrench')
weapons = [knife, revolver, wrench]

symbols = characters + rooms + weapons

def check_knowledge(knowledge):
    for symbol in symbols:
        if model_check(knowledge,symbol):
            termcolor.cprint(f'{symbol}: YES','green')
        elif not model_check(knowledge,Not(symbol)):
            termcolor.cprint(f'{symbol}: MAYBE','yellow')

knowledge=And(
    Or(mustard,pulm,scarlet),
    Or(ballroom,kitchen,library),
    Or(knife,revolver,wrench)
)
check_knowledge(knowledge)
print()

knowledge.add(
    And(Not(mustard),Not(kitchen),Not(revolver))
)
check_knowledge(knowledge)
print()

knowledge.add(
    Or(Not(library),Not(scarlet),Not(wrench)))
check_knowledge(knowledge)
print()

knowledge.add(Not(pulm))
check_knowledge(knowledge)
print()

knowledge.add(Not(ballroom))
check_knowledge(knowledge)
print()