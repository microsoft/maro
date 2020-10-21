#ifndef _MARO_BACKEND_RAW_BACKEND
#define _MARO_BACKEND_RAW_BACKEND

#include "common.h"
#include "attribute.h"

/*
Backend used to group attributes into nodes, as Node is a concept here.

Basically backend contains 2 part:

1. current frame:

Used to hold latest attribute information, writable for outside.

current frame is consist with 2 parts:

1). changale area

2). un-changable area


2. snapshot list (only contains attributes that will be in snapshot):

Use to save list of snapshots for current frame (changable attributes), this would be readly only for outside



POSSIABLE STRUCTURES:

1. fixed


|--------------------- current frame ------------------------------|
|              changable area               |  un-changable area   |
|    node 1            |         node 2-----|     node 1           |
|-------------------------------- ----------|----------------------|
| a11 | a12 | a2 | a3 | a4 | a5 | a61 | a62 | a7 | a81 | a82 | a83 |

|----------------- snapshot 1 --------------|
|               changable area              |
|    node 1            |         node 2     |
|-------------------------------------------|
| a11 | a12 | a2 | a3 | a4 | a5 | a61 | a62 |

|----------------- snapshot N --------------|
|               changable area              |
|    node 1            |         node 2     |
|-------------------------------------------|
| a11 | a12 | a2 | a3 | a4 | a5 | a61 | a62 |


2. dynamic


|---------------------------------------------- current frame ---------------------------------------------|
| n1-a11 | n1-a12 | u-n2-a2 | n1-a3 | n2-a4 | n2-a5 | n3-a61 | n3-a62 | u-n3-a7 | n1-a81 | n1-a82 | n1-a83 |


| --------------------------------------- snapshot 1 ----------------------------------|
| n1-a11 | n1-a12 | n1-a3 | n2-a4 | n2-a5 | n3-a61 | n3-a62 | n1-a81 | n1-a82 | n1-a83 |

| --------------------------------------- snapshot N ----------------------------------|
| n1-a11 | n1-a12 | n1-a3 | n2-a4 | n2-a5 | n3-a61 | n3-a62 | n1-a81 | n1-a82 | n1-a83 |


3. complex


|------------------------------- current frame ----------------------------------------------|
|-------------------------|----------- changable area --------------- |  un-changable area   |
|-------------------------|      node 1         |         node 2      |     node 1           |
|      node indices       |-------------------------------------------|----------------------|
| n1-indices | n2-indices | a11 | a12 | a2 | a3 | a4 | a5 | a61 | a62 | a7 | a81 | a82 | a83 |


|------------------------------- snapshot 1 --------------------------|
|-------------------------|------------------ changable area ---------|
|-------------------------|    node 1            |         node 2-----|
|  node indices           |-------------------------------- ----------|
| n1-indices | n2-indices | a11 | a12 | a2 | a3 | a4 | a5 | a61 | a62 |


|--------------------------------snapshot N------------------------------------------------|
|-------------------------|--------------------- changable area ---------------------------|
|-------------------------|      node 1         |         node 2      |         node 3     |
|-----node indices--------|----------------------------------------------------------------|
| n1-indices | n2-indices | a11 | a12 | a2 | a3 | a4 | a5 | a61 | a62 | a7 | a8 | a9 | a10 |


To support dynamic nodes (add/remove), we have to method:

1. No in-memory snapshot over-write
 
 do not support snapshot overwrite, just keep all snapshot in mem-mapping file. we can just expend the file to support more data


2. with in memory snapshot over-write
 we need another mapping that used to index every part of snapshot, as the length will be different for snapshots.

 When over-writing:
 1). if existing snapshot is larger than current one, then we just save current frame in current snapshot, and leave the additional space there
 2). if existing snapshot is shorter than current one, then we have 2 way to due with:
    1)). split current frame into 2 parts, 1st one's length same as over-writing snapshot, 2nd append to the end (after allocate new memory)
    2)). just allocate a large enough meomry to hold this, make over-writing snapshot as avaialble


all above methods need a table to track avaiable and existing snapshot, but may be the content is different
|                      snapshot mapping table                             |





*/


namespace maro
{
    namespace backends
    {
        namespace raw
        {
          class Backend
          {
          };
        } // namespace raw
    }     // namespace backends
} // namespace maro

#endif
