/**
 * 
 * currently each bin file only contains only one stock info, and can be easily update (append to the end)
 * 
 * structure of file
 * 
 * 
 * # meta part
 * 4 bytes (char): MARO
 * 1 byte (unsigned): data type (stock=1, futures=2, ...), used to identify usage of file
 * 1 byte (unsigned): version of this file, used for compact issue
 * 2 bytes (unsigned): size of each item
 * 4 bytes (unsinged): id of this type
 * 4 bytes (unsinged): date time of first item (seconds since 1970)
 * 4 bytes (unsinged): date time of last item (seconds since 1970)
 * 
 * # data part, that match (size of each item) * (count of items)
 * 
 * NOTE: data can be indiced by number and item size, and we should fill with empty if we missing some item
 * 
 * 
 * **/

#ifndef _CONVERTER_H
#define _CONVERTER_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <time.h>
#include <math.h>


// TODO: may not compact with Windows now.
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "jsmn.h"

#define BOOL uint8_t
#define FALSE 0
#define TRUE 1

const int64_t DAY_SEC = 24 * 60 * 60;

typedef enum{
    VALID_STOCK = 0,
    INVALID_STOCK = 1
} stock_state_e;


// type of entities we support
typedef enum{
    CONV_STOCK = 1,
    CONV_FUTURES = 2
} dtype_e;

typedef struct Meta{
    char header[5];
    char id[7];
    uint8_t dtype;
    uint8_t version;
    uint16_t item_size;
    
    uint64_t start_time;
    uint64_t end_time;
    // we can get item count byte (file size - size of meta)/item size
} meta_t;

typedef struct Stock{
    uint8_t is_valid; // is this item is valid, 0=valid, 1=invalid
    float opening_price;
    float closing_price;
    float pre_closing_price; // close price of last day
    float highest_price;
    float lowest_price;
    float up_down_amount;
    float up_down_rate;
    float turnover_rate;
    float trade_amount;
    float total_market_capitalization;
    float circulation_market_capitalization;
    uint32_t trade_volume;
    uint32_t trade_num;
    uint64_t time; // seconds since 1970

    //idr
    float daily_return;
    char code[7]; // code name

} stock_t;

typedef struct FinReader{
    int fd; // file descriptor
    int size; // item count
    void *addr; // address to the memory to read (from mmap)
    meta_t meta;
    int start; // start index for query
    int cur_index; // current index, used for continue quering
    uint8_t dtype; // which item type this reader for (like stock, futures)
    void *data; // current result of quering
} finreader_t;

/*
create a new output file, and fill the meta part 
*/

void new_stock_bin(int8_t version, char code[6], const char *src_path, const char *output_path);
void append_stock_bin(const char *src_path, const char *output_path);

// convert input datetime into seconds since 1970
time_t get_time(const char *datetime);

// read all the content from target file
char *read_json(const char *path);

// parse and return list of tokens
jsmntok_t *parse_json(const char *json, int *tsize);

// get index object item (NOTE: for this util only)
int next_object_token(const char *json, jsmntok_t *tokens, int tsize, int start_index);

inline void read_property(const char *json, jsmntok_t *tok, char *buffer);

// read and fill the stock info
void read_stock_from_json(const char *json, jsmntok_t *tokens, int start_index, stock_t *stock);

void write_meta(FILE *file, meta_t *meta);

void read_meta(FILE *file);

// write stock into file, padding if current_day - last_day > 1
// return 0 if success or 1
int write_stock_item(FILE *file, stock_t *stock, uint64_t last_time);//


/**
 * interfaces to read binary file with mmap
 * **/

BOOL init_reader(const char *path, finreader_t *reader, int8_t dtype);
void release_reader(finreader_t *reader);
void reset_reader(finreader_t *reader);

// read and fill next item
BOOL next_item(finreader_t *reader);
BOOL peek_item(finreader_t *reader);
BOOL step_to_next(finreader_t *reader);

// reader next stock item, and move the pointer to next
BOOL next_stock_item(finreader_t *reader);
// void next_futures_item(finreader_t *reader, futures_t *future);

// peek data at current pointer, will not change the poitner
BOOL peek_stock_item(finreader_t *reader);


void cal_stock_daily_return(stock_t *stock);



/***** for data combination ******/

#define MIN_COMBINE_ARGUMENT_NUM 8

typedef struct CombineHeader{
    uint16_t item_length; // length of each item
    uint32_t item_number; // number of all items
    uint32_t steps; 
    uint64_t start_time;
    uint64_t end_time;

} combine_header_t;

typedef struct CombineRowMeta{
    uint16_t item_number;
    uint32_t tick;
} combine_row_meta_t;


typedef struct CombineWriter
{
    FILE *file;
    combine_header_t header;
} combine_writer_t;

typedef struct CombineReader
{
    void *addr;
    stock_t *buffer; // size same as item number in meta
    int fd;
    int current_row_length; // stocks in current row
    uint32_t current_tick;
    size_t size;
    size_t offset; // current offset
    combine_header_t *meta;
} combine_reader_t;


void init_combination_writer(char *path, combine_writer_t *writer, int64_t start_time, int64_t end_time, int16_t item_number, int16_t steps);
void release_combination__writer(combine_writer_t *writer);

void process_combination(char *ouput_path, uint64_t start_time, uint64_t end_time, uint32_t steps, int items, char *item_path[]);

void new_combination_row(combine_writer_t *writer, uint32_t time);
void update_combination_item_number(combine_writer_t *writer, int16_t item_number);
void add_combination_stock(combine_writer_t *writer, stock_t *stock);


void init_combination_reader(char *path, combine_reader_t *reader);
void release_combination_reader(combine_reader_t *reader);
BOOL peek_current_row_info(combine_reader_t *reader, uint16_t *number, uint32_t *tick);
int read_combination_row(combine_reader_t *reader); // reader stocks same with stock number in current row, return stock number
stock_t* read_combination_item(combine_reader_t *reader, int index); // reader item by index (less than number from read_combination_row)
void reset_combination_reader(combine_reader_t *reader);



#endif