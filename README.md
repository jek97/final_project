# final_project
implementation of the basic Dyna-q for the solution of the problem

Update: used the professor valid action method.

Update: improved the plotting part as well as the algorithm itself

Update: joined the Q-factors and the model in a single data structure, also implemented the prioritized sweeping

Update: functions compacted/optimized

Update: modified the data structure to grow while we explore new state/actions, moreover moved to the dyna-q+

Update: found an error, i was modifing the model even in simulation, instead of updating the Q

Update: tring to modify the model in a single dict, check functions and order of the arguments, it may be wrong sometimes.

Update: made a single dict, much faster, thinking about create one dict for the q and one for the model, also created a bell shaped function if i want ot vary eps and alpha along the episodes.

Update: i will modify the code for expand the model only when we actually visit a state
