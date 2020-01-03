Resource Allocator
==================

The resource allocator is responsible for allocating hardware resources to distributed components.
For the current version, it suffices to use manual (e.g., configuration-file-based) allocation
schemes. So you may specify the number of cpu cores to each component in a configuration file.
However, as a distributed system involves more and more machines, manual allocation becomes tedious
and error prone and it may become necessary to design and implement an efficient dynamic resource
allocator.