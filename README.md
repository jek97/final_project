# final_project
implementation of the basic Dyna-q for the solution of the problem

Update: used the professor valid action method.

Update: improved the plotting part as well as the algorithm itself

Update: joined the Q-factors and the model in a single data structure, also implemented the prioritized sweeping

Update: functions compacted/optimized

Update: modified the data structure to grow while we explore new state/actions, moreover moved to the dyna-q+

Update: found an error, i was modifing the model even in simulation, instead of updating the Q

Update: tring to modify the model in a single dict, check functions and order of the arguments, it may be wrong sometimes.

Update: made a single dict, much faster, thinking about create one dict for the q and one for the model, also created a bell shaped function if i want to vary eps and alpha along the episodes.

Update: modified to actually expand the model only with states/actions encountered. also found some error in simulation phase with the states passed to the functions.

Update: found some error in the q_update, i will test with a changing parameters along the way.
