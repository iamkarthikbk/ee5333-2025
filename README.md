# ee5333-2025: Intro to Physical Design Automation

## Contents of this repository

1) Code snippets
2) Assignments
3) Tutorials

## Course Information

Course Instructor: Prof. Ramprasath S <<ramprasath@ee.iitm.ac.in>>

1) Introduction: Motivating examples, basics of graph theory (Revise/introduce internals of transistor level layouts, Steps involved in the physical design flow, demo complexity of layout generation using industrial layouts, netlists as graphs/hypergraphs; Assignment: parse netlist and represent as graphs/hypergraphs.)

2) Partitioning: Kernighan-Lin, Fiduccia-Mattheyses algorithms, and multilevel partitioning (Introduce divide-and-conquer paradigm, partitioning as a follow-through, complexity of partitioning; various classes of algorithms to partition; Assignment: Implement the partitioning algorithms and compare results for benchmark circuits)

3) Floorplanning: Representations, constraint graphs, slicing, non-slicing, sequence-pairs, simulated annealing and pin assigment. (Floorplanning objective, various abstract representations of floorplans, transforming representations to constraint graphs, solving constraint graphs to realize concrete floorplans, optimization of floorplans for estimated area and wire length using simulated annealing; Assignment: generate floorplans for previously partitioned circuits and assign ports)

4) Placement: Digital and AMS placement; Integer Linear Program (ILP) and simulated annealing placers, min-cut and analytic (quadratic and force-directed) global placement, legalization and detailed placement; (Complexity of placement, Need for global and detailed placement, various placement algorithms; Assignment: Place cells in each of the blocks in previous floorplan)

5) Routing: Single-net routing: spanning trees, shortest path routing using Dijkstra and A*, ILP routing; handling multi-net routing, channel routing, clock-tree synthesis, symmetric, length-constrained and octilinear routing (Problems in routing, Introduce nets, wires, and constituent routing shapes; algorithms to connect pins using shortest path routes; handling conflicts in routing multiple nets; introduce routing of special nets; Assignment: implement shortest path and global routing)

6) Design rule checks: Commonly encountered DRCs, Boolean operations on polygons and region query. (Complexity in physical design arising from advanced manufacturing; design rules from foundries and their sources; boolean operations on polygons to verify design rule compliance; region query for faster verification;

Assignment: Implement a DRC checker for a given set of rules)
Programming assignments will be in Python/C/C++.
Prerequisite: Rudimentary programming experience in either Python/C/C++