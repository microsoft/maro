/**
 * 
 * header:
 * item_number: 4 bytes
 * item_length: 2 byte
 * start_time: 8 bytes
 * end_time: 8 bytes
 * 
 * data (by row):
 * time: 8 byte (time of current row)
 * item number: number of items in current row
 * items: current_time_item_number * item_length
 * 
 * **/

#ifndef _COMBINE_H
#define _COMBINE_H


#include "converter.h"



typedef struct CombineHeader{
    int32_t item_number; // number of all items
    int16_t item_length; // length of each item
    int64_t start_time;
    int64_t end_time;

} combine_header_t;

typedef struct CombineRowMeta{
    int16_t item_number;
    int64_t time;
} combine_row_meta_t;


typedef struct CombineWriter
{
    FILE *file;
    combine_header_t header;
} combine_writer_t;

typedef struct CombineReader
{
    int fd;
    void *addr;
} combine_reader_t;


void init_writer(char *path, combine_writer_t *writer, int64_t start_time, int64_t end_time, int16_t item_number);
void release_writer(combine_writer_t *writer);

void process(char *root, char *item_names[]);

void new_row(combine_writer_t *writer, int64_t time);
void update_item_number(combine_writer_t *writer, int16_t item_number);
void add_stock(combine_writer_t *writer, stock_t *stock);

#endif