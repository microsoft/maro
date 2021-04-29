# Offline Integer Linear Programming (ILP) For VM Scheduling

## Data used in the formulation
The offline lp example directly uses the future VM information to formulate the future machine resource changes.
The data the formulation is based is extracted from the business engine directly (the data accessing way is not workable for online methods, just to act as a baseline method), including:
- the capacity (#core, #memory) of all the PMs
- the arrival time of the VM request
- the resource requirement (#core, #memory) of the VM request
- the lifetime of the VM
- all the existing VM-PM mapping 

## ILP configuration
Based on the data and the configuration for the problem formulation, the ILP solver will calculate to get an allocation solution. The configuration includes:
- **solver**: The solver used, CBC is the default solver for PuLP. While GLPK is another free solver that it can get more stable solution but need additional installation of glpk-utils.
- **plan_window_size**: How many ticks the formulation will take into account.
- **apply_buffer_size**: For each solution, the allocation of how many ticks will be applied. E.g. Assume the plan_window_size is 10 and the apply_buffer_size is 2. And assume that at tick 2, we trigger a formulation. Then formulation will simulate the status changes from tick 2 to tick 11 (plan window size is 10), and get an allocation plan for each VM requests in this time window. But only the allocation of VM requests during tick 2 to tick 3 (apply buffer size is 2) will be applied according to the solution. The first VM request comes at tick 4 will trigger a new formulation and get a new solution...
- **core_safety_remaining_ratio**: Used in online case, used to maintain a safety inventory. It is enough to set to 0 in offline case.
- **mem_safety_remaining_ratio**: Used in online case, used to maintain a safety inventory. It is enough to set to 0 in offline case.
- **successful_allocation_decay**: The decay factor of each successful allocation in the objective of the formulation. To be specific, a successful allocation for the first request in the formulation will counts 1, the one for the second request in the formulation will counts 1 * decay, the one for the third request in the formulation will counts 1 * decay * decay, ...
- **allocation_multiple_core_num**: If set up, the base of a successful allocation will be set up as #core the request required instead of 1.
- **dump_all_solution**: Whether to dump the lp formulation and solution to file or not.
- **dump_infeasible_solution**: If there exist infeasible solutions, whether to dump them to file or not.
- **stdout_solver_message**: Whether to enable the solver message output or not. Suggest to disable it if the solution spends not too much time.

*Notes: For large problem scale, the free solver CBC and GLPK may take a long time to get the solution. Sometimes they even cannot handle large problem scale. We suggest you to use them only for small problem scale or buy some powerful solver if needed.*