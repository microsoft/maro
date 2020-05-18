#include "converter.h"


/**
 * usage:
 * converter datatype mode options_for_each_mode
 *
 * datatype:
 * 1. stock
 * 
 * mode:
 * 1. new
 * 2. update
 * 
 * convert should support 2 mode:
 * 1. new (override): convert and write specified json file into binary
 *    converter stock new id ver data.json data_1.bin
 * 2. update (append): convert and append json file into specified binary file, and update the item count 
 *    converter stock append data.json data_1.bin
 * 3. combine : combine specified bin files into one, with specified start and end time
 *    converter stock combine start_time end_time src_folder 00001.bin 00002.bin 00003.bin
 * **/

int main(int argc, char *argv[])
{
    // combine_reader_t reader;

    // init_combination_reader(argv[1], &reader);

    // int l = read_combination_row(&reader);
    // l = read_combination_row(&reader);
    
    // l = read_combination_row(&reader);
    // l = read_combination_row(&reader);
    // l = read_combination_row(&reader);
    // l = read_combination_row(&reader);
    // l = read_combination_row(&reader);
    // l = read_combination_row(&reader);
    // l = read_combination_row(&reader);
    // l = read_combination_row(&reader);
    // l = read_combination_row(&reader);

    // release_combination_reader(&reader);

    // return 1;

    if(argc < 3){
        char *help = "converter command line to convert finance json data into binary format, basic usage:\n\
\n\
  Stock: support 2 modes: 'new' and 'append'\n\
    . new: create or overwrite the specified file with input json data.\n\
      converter stock new stock_id file_version json_path output_bin_path\n\
    . append: append input json data into exists binary file \n\
      converter stock append json_path bin_path\n\
    ";

        printf("%s.\n", help);

        return 1;
    }

    char *dtype = argv[1];
    char *mode = argv[2];

    if(strcmp(dtype, "stock") == 0)
    {
        uint32_t id = atoi(argv[3]);

        printf("dtype: %s, mode: %s.\n", dtype, mode);

        if(strcmp(mode, "new") == 0)
        {
            if(argc < 6)
            {
                printf("invalid number of arguments for mode 'new'.\n");

                return 1;
            }

            int8_t ver = atoi(argv[4]);
            char *src_path = argv[5];
            char *output_path = argv[6];

            new_stock_bin(ver, id, src_path, output_path);
        }
        else if(strcmp(mode, "update") == 0){
            char *src_path = argv[3];
            char *output_path = argv[4];

            append_stock_bin(src_path, output_path);
        }
        else if(strcmp(mode, "combine") == 0)
        {
            char *output_path = argv[3];
            int64_t start_timestamp = atoi(argv[4]);
            int64_t end_timestamp = atoi(argv[5]);
            int32_t steps = atoi(argv[6]);
            int items = argc - MIN_COMBINE_ARGUMENT_NUM + 1;

            char** items_path = malloc(sizeof(char*) * items);

            for(int i=0;i<items;i++)
            {
                items_path[i] = argv[i + MIN_COMBINE_ARGUMENT_NUM - 1];
            }

            // printf("%s.\n", items_path[0]);
            process_combination(output_path, start_timestamp, end_timestamp, steps, items, items_path);

            free(items_path);
        }
    }

    return 0;
}

void write_meta(FILE *file, meta_t *meta)
{   
    fseek(file, 0, SEEK_SET);
    fwrite(meta, sizeof(meta_t), 1, file);
}

void get_meta(FILE *file, meta_t *meta)
{
    fseek(file, 0, SEEK_SET);

    fread(meta, sizeof(meta_t), 1, file);
}

void append_stock_bin(const char *src_path, const char *output_path)
{
    meta_t meta;
    FILE *file = fopen(output_path, "rb+");

    if(file != NULL)
    {
        get_meta(file, &meta);
        
        // seek to the end for append
        fseek(file, 0, SEEK_END);

        // printf("id: %d, dtype: %d, version: %d, item size: %d, start: %lld, end: %lld. \n",\
        //         meta.id, meta.dtype, meta.version, meta.item_size, meta.start_time, meta.end_time);
    
        char *json = read_json(src_path);

        if(json != NULL)
        {
            int tsize = 0;
            jsmntok_t *tokens = parse_json(json, &tsize);

            if(tokens != NULL)
            {
                int tindex = 1;
                uint64_t last_time = meta.end_time;
                jsmntok_t *tok = NULL;
                stock_t stock;

                // try to find token of next object 
                while((tindex = next_object_token(json, tokens, tsize, tindex)) != -1)
                {
                    tok = tokens + tindex;

                    read_stock_from_json(json, tokens, tindex, &stock);
                    
                    if(write_stock_item(file, &stock, last_time) == 0)
                    {
                        last_time = stock.time;
                    }

                    // NOTE: we assuming that out input json have no nested object
                    tindex += tok->size * 2;
                }

                meta.end_time = last_time;

                write_meta(file, &meta);

                free(tokens);
            }

            free(json);
        }
    }
    else
    {
        printf("Fail to open file %s.\n", output_path);
    }
    

    fclose(file);
}

void new_stock_bin(int8_t version, int32_t code, const char *src_path, const char *output_path)
{
    meta_t meta = {"MARO", CONV_STOCK, version, sizeof(stock_t), code, 0, 0};
    
    FILE *file = fopen(output_path, "wb+");

    // write meta first
    write_meta(file, &meta);

    char *json = read_json(src_path);
    int tsize = 0;

    if (json != NULL)
    {
        jsmntok_t *tokens = parse_json(json, &tsize);

        if (tokens != NULL)
        {   
            int tindex = 1; // we skip 0 index, as it is the root
            int count = 0; // count of items we find
            uint64_t last_time = 0;
            jsmntok_t *tok = NULL;
            stock_t stock;

            while((tindex = next_object_token(json, tokens, tsize, tindex)) != -1)
            {
                count++;
                tok = tokens + tindex;

                read_stock_from_json(json, tokens, tindex, &stock);

                if(write_stock_item(file, &stock, last_time) == 0)
                {
                    last_time = stock.time;
                }

                // NOTE: we assuming that out input json have no nested object
                tindex += tok->size * 2;

                if(count == 1)
                {
                    // if it is first item we find, then we should update the time in meta

                    meta.start_time = stock.time;
                    last_time = stock.time;
                }
            }

            meta.end_time = last_time;

            write_meta(file, &meta);

            free(tokens);
        }

        free(json);
    }

    fclose(file);
}

jsmntok_t *parse_json(const char *json, int *tsize)
{
    jsmn_parser p;

    jsmn_init(&p);

    // first time we parse the json to get tokens count
    *tsize = jsmn_parse(&p, json, strlen(json), NULL, 0);

    // allocate memory for tokens
    jsmntok_t *tokens = (jsmntok_t *)malloc(sizeof(jsmntok_t) * (*tsize));

    // reset and parse it again, this time the result will be saved into tokens
    jsmn_init(&p);
    jsmn_parse(&p, json, strlen(json), tokens, *tsize);

    return tokens;
}

int next_object_token(const char *json, jsmntok_t *tokens, int tsize, int start_index)
{
    jsmntok_t *tok = NULL;

    for (int i = start_index; i < tsize;i++)
    {
        tok = tokens + i;

        if(tok->type == JSMN_OBJECT)
        {
            return i;
        }
    }

    return -1; // -1 means not found any object after start index
}

void read_property(const char *json, jsmntok_t *tok, char *buffer)
{
    // reset the buffer to make sure it contains \0 at the end of string
    memset(buffer, 0, strlen(buffer));

    strncpy(buffer, json + tok->start, tok->end - tok->start);
}

int write_stock_item(FILE *file, stock_t *stock, uint64_t last_time)
{
    fseek(file, 0, SEEK_END);

    cal_stock_daily_return(stock);

    fwrite(stock, sizeof(stock_t), 1, file);

    return 0;
}

void read_stock_from_json(const char *json, jsmntok_t *tokens, int start_index, stock_t *stock)
{   
    stock->is_valid = VALID_STOCK;

    char buffer[1024];

    // date
    read_property(json, tokens + start_index + 2, buffer);

    stock->time = (uint64_t)get_time(buffer);

    // code
    read_property(json, tokens + start_index + 4, buffer);

    stock->code = atoi(buffer);

    // close price    
    read_property(json, tokens + start_index + 8, buffer);

    stock->closing_price = atof(buffer);

    // highest price    
    read_property(json, tokens + start_index + 10, buffer);

    stock->highest_price = atof(buffer);

    // lowest_price    
    read_property(json, tokens + start_index + 12, buffer);

    stock->lowest_price = atof(buffer);

    // opening_price    
    read_property(json, tokens + start_index + 14, buffer);

    stock->opening_price = atof(buffer);

    // pre_closing_price    
    read_property(json, tokens + start_index + 16, buffer);

    stock->pre_closing_price = atof(buffer);

    // up_down_amount    
    read_property(json, tokens + start_index + 18, buffer);

    stock->up_down_amount = atof(buffer);

    // up_down_amount     
    read_property(json, tokens + start_index + 20, buffer);

    stock->up_down_rate = atof(buffer);

    // turnover_rate
    read_property(json, tokens + start_index + 22, buffer);

    stock->turnover_rate = atof(buffer);

    // trade_volume 
    read_property(json, tokens + start_index + 24, buffer);

    stock->trade_volume = atoi(buffer);

    //trade_amount
    read_property(json, tokens + start_index + 26, buffer);

    stock->trade_amount = atof(buffer);

    //total_market_capitalization
    read_property(json, tokens + start_index + 28, buffer);

    stock->total_market_capitalization = atof(buffer);

    //circulation_market_capitalization
    read_property(json, tokens + start_index + 30, buffer);

    stock->circulation_market_capitalization = atof(buffer);

    //trade_num
    read_property(json, tokens + start_index + 32, buffer);

    stock->trade_num = atoi(buffer);
}

time_t get_time(const char *date)
{
    struct tm c_time;

    strptime(date, "%Y-%m-%d", &c_time);

    // reset hour, min, sec as we should only have y, m and d
    c_time.tm_hour = 0;
    c_time.tm_min = 0;
    c_time.tm_sec = 0;
    c_time.tm_isdst = 0;

    time_t t = mktime(&c_time);
    
    return t;
}

// DO NOTE to free the content
char *read_json(const char *path)
{
    FILE *fp = fopen(path, "r");

    if (fp == NULL)
    {
        return NULL;
    }

    // to the end
    fseek(fp, 0, SEEK_END);

    long fsize = ftell(fp);

    char *json = (char *)calloc(fsize + 1, sizeof(char));

    if (json != NULL)
    {
        fseek(fp, 0, SEEK_SET);
        fread(json, sizeof(char), fsize, fp);
    }

    fclose(fp);

    return json;
}


BOOL init_reader(const char *path, finreader_t *reader, int8_t dtype)
{   
    // open the file to get the file descripter
    reader->fd = open(path, O_RDONLY, 0);
    
    if(reader->fd == -1){
        printf("%s\n", path);
        perror("Fail to open file.");

        return FALSE;
    }

    struct stat st;

    stat(path, &st);
    
    reader->addr = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, reader->fd, 0);
    
    if(reader->addr == MAP_FAILED)
    {
        perror("Fail to map the file.");

        return FALSE;
    }

    // read the meta 
    reader->meta = *((meta_t *)reader->addr);

    reader->size = (st.st_size - sizeof(meta_t)) / reader->meta.item_size;

    reader->data = NULL;

    if(dtype == CONV_STOCK)
    {
        reader->dtype = dtype;
    }

    reader->cur_index = reader->start;

    return TRUE;
}

void reset_reader(finreader_t *reader){
    if (reader == NULL) return;

    reader->cur_index = reader->start;
}

void release_reader(finreader_t *reader){
    if(reader == NULL) return;

    munmap(reader->addr, reader->size);

    reader->addr = NULL;

    close(reader->fd);

    reader->data = NULL;
}

BOOL next_item(finreader_t *reader)
{
    if(reader == NULL) return FALSE;

    if(reader->dtype == CONV_STOCK)
    {
        return next_stock_item(reader);
    }

    return FALSE;
}

BOOL peek_item(finreader_t *reader)
{
    if(reader == NULL) return FALSE;

    if(reader->dtype == CONV_STOCK)
    {
        return peek_stock_item(reader);
    }

    return FALSE;
}

BOOL step_to_next(finreader_t *reader)
{
    if(reader==NULL || reader->cur_index >= reader->size) return FALSE;

    reader->cur_index += 1;

    return TRUE;
}

BOOL next_stock_item(finreader_t *reader)
{
    if(TRUE == peek_stock_item(reader))
    {
        return step_to_next(reader);
    }

    return FALSE;
}

// peek data at current pointer, will not change the poitner
BOOL peek_stock_item(finreader_t *reader)
{
    if(reader == NULL) return FALSE;

    if(reader->cur_index >= reader->size)
    {   
        // if the query out of the boundary, then return last result
        return FALSE;
    }

    stock_t *data_ptr = (stock_t*)(reader->addr + sizeof(meta_t));

    reader->data = data_ptr + (reader->cur_index);

    return TRUE;
}



/**********************************************************/
void cal_stock_daily_return(stock_t *stock)
{
    if(stock == NULL) return;

    if(stock != NULL && stock->is_valid == VALID_STOCK && stock->pre_closing_price != 0)
    {
        stock->daily_return = (stock->closing_price / stock->pre_closing_price) - 1;

        printf("c: %f, pc: %f, d: %f.\n", stock->closing_price, stock->pre_closing_price, stock->daily_return);
    }
}



/*** data combination ***/

void process_combination(char *ouput_path, uint64_t start_time, uint64_t end_time, uint32_t steps, int items, char *item_path[])
{
    // init writer
    combine_writer_t writer;

    init_combination_writer(ouput_path, &writer, start_time, end_time, items, steps);

    // init readers
    finreader_t *readers = (finreader_t*)calloc(items, sizeof(finreader_t));

    if(readers == NULL)
    {
        perror("Fail to init readers");

        return;
    }

    BOOL is_init_success = TRUE;

    for (int i=0;i<items;i++)
    {
        if (FALSE == init_reader(item_path[i], &readers[i], CONV_STOCK))
        {
            perror("fail to open file");

            is_init_success = FALSE;

            break;
        }

        // printf("reader start time: %llu, end time: %llu.\n", readers[i].meta.start_time, readers[i].meta.end_time);
    }

    // read and combine into new file
    if(TRUE == is_init_success)
    {
        finreader_t *reader;
        stock_t *stock=NULL;
        uint64_t cur_time = start_time;
        uint16_t row_items_number = 0;
        uint32_t tick = 0;

        // printf("start time: %llu, end time: %llu, steps: %d.\n", start_time, end_time, steps);
        
        while (cur_time < end_time)
        {
            row_items_number = 0;

            // add row meta
            new_combination_row(&writer, tick);

            for(int i=0;i<items;i++)
            {
                reader = &readers[i];
                
                if(TRUE == peek_stock_item(reader))
                {
                    do
                    {
                        stock = (stock_t*)reader->data;

                        if(stock->time >= cur_time)
                        {
                            break;
                        }
                        
                        step_to_next(reader);
                    } while (peek_stock_item(reader) == TRUE);

                    if( stock->time >= cur_time && (stock->time - cur_time) < steps)
                    {
                        // move the reader pointer
                        step_to_next(reader);

                        // add stock item to export file
                        add_combination_stock(&writer, stock);

                        row_items_number++;
                        
                        // printf("find stock, time: %llu <==> %llu, %llu.\n", stock->time, cur_time, (stock->time - cur_time));

                        step_to_next(reader);
                    }
                }
            }   

            if(row_items_number > 0)
            {
                update_combination_item_number(&writer, row_items_number);
            }

            cur_time += steps;
            tick += 1;
        }
    }

    if(readers != NULL)
    {
        for(int i=0;i<items;i++)
        {
            release_reader(&readers[i]);
        }

        free(readers);
    }
}


void init_combination_writer(char *path, combine_writer_t *writer, int64_t start_time, int64_t end_time, int16_t item_number, int16_t steps)
{
    writer->file = fopen(path, "wb+");

    combine_header_t t = {sizeof(stock_t), item_number, steps, start_time, end_time};

    writer->header = t;

    fwrite(&t, sizeof(t), 1, writer->file);
}

void release_combination__writer(combine_writer_t *writer)
{
    if(writer != NULL)
    {
        if(writer->file != NULL)
        {
            fclose(writer->file);

            writer->file = NULL;
        }
    }
}

void new_combination_row(combine_writer_t *writer, uint32_t tick)
{
    if(writer == NULL)
    {
        return;
    }

    combine_row_meta_t row_meta = {0, tick};

    fwrite(&row_meta, sizeof(combine_row_meta_t), 1, writer->file);
}

void update_combination_item_number(combine_writer_t *writer, int16_t item_number)
{
    if(writer == NULL && item_number == 0)
    {
        return;
    }

    fseek(writer->file, -(writer->header.item_length * item_number + sizeof(combine_row_meta_t)), SEEK_END);

    fwrite(&item_number, sizeof(int16_t), 1, writer->file);

    fseek(writer->file, 0, SEEK_END);
}

void add_combination_stock(combine_writer_t *writer, stock_t *stock)
{
    if(writer == NULL)
    {
        return;
    }

    // printf("new item: time: %llu\n", stock->time);

    fwrite(stock, sizeof(stock_t), 1, writer->file);
}

void init_combination_reader(char *path, combine_reader_t *reader)
{
    reader->fd = open(path, O_RDONLY, 0);
    
    if(reader->fd == -1){
        printf("%s\n", path);
        perror("Fail to open file.");
    }

    struct stat st;

    stat(path, &st);
    
    reader->size = st.st_size;
    reader->addr = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, reader->fd, 0);

    reader->meta = (combine_header_t*)reader->addr; // reader meta
    reader->offset += sizeof(combine_header_t); // update offset to row data
    reader->buffer = (stock_t*)calloc(reader->meta->item_number, sizeof(stock_t));

    // printf("size of file: %lld\n", st.st_size);

    // printf("item length: %d, item number: %d, steps: %d, start: %llu, end: %llu.\n", 
    //     reader->meta->item_length,
    //     reader->meta->item_number,
    //     reader->meta->steps,
    //     reader->meta->start_time,
    //     reader->meta->end_time);
}

void release_combination_reader(combine_reader_t *reader)
{
    if(reader != NULL)
    {
        if(reader->buffer != NULL)
        {
            free(reader->buffer);
        }

        reader->buffer = NULL;

        munmap(reader->addr, reader->size);

        reader->addr = NULL;
        reader->meta = NULL;
    }
}

int read_combination_row(combine_reader_t *reader)
{
    if(reader == NULL || reader->offset >= reader->size) return 0;

    // printf("offset: %zu\n", reader->offset);
    // printf("row meta size: %zu.\n", sizeof(combine_row_meta_t));
    
    combine_row_meta_t *r_meta = reader->addr + (reader->offset);

    reader->offset += sizeof(combine_row_meta_t);
    reader->current_row_length = r_meta->item_number;
    reader->current_tick = r_meta->tick;

    if(r_meta->item_number > 0)
    {
        // copy the data into our buffer
        memcpy((char*)(reader->buffer), (reader->addr + reader->offset), sizeof(stock_t));


        // stock_t *stock = NULL;

        // printf("current tick: %d\n", r_meta->tick);
        // for(int i=0;i<r_meta->item_number;i++)
        // {
        //     stock = reader->buffer + i;
        //     printf("time: %llu, opening price: %f\n", stock->time, stock->opening_price);
        // }
    }

    reader->offset += r_meta->item_number * (reader->meta->item_length);

    // printf("row, item number: %d, time: %llu.\n", r_meta->item_number, r_meta->time);
    
    return r_meta->item_number;
}

stock_t* read_combination_item(combine_reader_t *reader, int index)
{
    if(reader == NULL || index >= reader->current_row_length)
    {
        return NULL;
    }

    return reader->buffer + index;
}

void reset_combination_reader(combine_reader_t *reader)
{
    if(reader == NULL) return;

    reader->offset = sizeof(combine_header_t);
    reader->current_row_length = 0;
    reader->current_tick = 0;
    
}