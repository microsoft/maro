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
 * **/

int main(int argc, char *argv[])
{
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
        }else{
            char *src_path = argv[3];
            char *output_path = argv[4];

            append_stock_bin(src_path, output_path);
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

        printf("id: %d, dtype: %d, version: %d, item size: %d, start: %lld, end: %lld. \n",\
                meta.id, meta.dtype, meta.version, meta.item_size, meta.start_time, meta.end_time);
    
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

    // first item always be valid
    if(last_time == 0)
    {
        cal_stock_daily_return(stock);

        fwrite(stock, sizeof(stock_t), 1, file);

        return 0;
    }

    int64_t delta = stock->time - last_time;
    
    // printf("last time: %lld, stock time: %lld, day sec: %lld.\n", last_time, stock->time, DAY_SEC);

    // invalid item, we just skip it
    if(last_time != 0 && delta < DAY_SEC)
    {
        struct tm * cur;
        char buffer[26];
        cur = localtime(&(stock->time));
        strftime(buffer, 26, "%Y:%m:%d %H:%M:%S", cur);

        struct tm * last;
        char last_buffer[26];
        last = localtime(&(last_time));
        strftime(last_buffer, 26, "%Y:%m:%d %H:%M:%S", last);
        // do nothing, just a warning.
        printf("last time: %lld, stock time: %lld, day sec: %lld.\n", last_time, stock->time, DAY_SEC);

        printf("warning: skip invliad item to insert, delta: %lld. time: %s. last_time: %s\n", delta, buffer, last_buffer);

        return 1;
    }

    // try to padding
    while(delta > 0)
    {
        delta -= DAY_SEC;
        last_time += DAY_SEC;
        stock->time = last_time;
        stock->is_valid = (delta == 0 ? VALID_STOCK : INVALID_STOCK);

        // printf("write time: %lld.\n", stock->time);

        cal_stock_daily_return(stock);
        fwrite(stock, sizeof(stock_t), 1, file);
    }

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

    // name + 6

    // close price    
    read_property(json, tokens + start_index + 14, buffer);

    stock->closing_price = atof(buffer);

    // highest price    
    read_property(json, tokens + start_index + 10, buffer);

    stock->highest_price = atof(buffer);

    // lowest_price    
    read_property(json, tokens + start_index + 12, buffer);

    stock->lowest_price = atof(buffer);

    // opening_price    
    read_property(json, tokens + start_index + 8, buffer);

    stock->opening_price = atof(buffer);

    // pre_closing_price    
    // read_property(json, tokens + start_index + 16, buffer);

    stock->pre_closing_price = 0;// atof(buffer);

    // up_down_amount    
    // read_property(json, tokens + start_index + 18, buffer);

    stock->up_down_amount = 0;// atof(buffer);

    // up_down_amount     
    // read_property(json, tokens + start_index + 20, buffer);

    stock->up_down_rate = 0;// atof(buffer);

    // turnover_rate
    // read_property(json, tokens + start_index + 22, buffer);

    stock->turnover_rate = 0;// atof(buffer);

    // trade_volume 
    read_property(json, tokens + start_index + 18, buffer);

    stock->trade_volume = atoi(buffer);

    //trade_amount
    // read_property(json, tokens + start_index + 26, buffer);

    stock->trade_amount = 0;// atof(buffer);

    //total_market_capitalization
    // read_property(json, tokens + start_index + 28, buffer);

    stock->total_market_capitalization = 0;// atof(buffer);

    //circulation_market_capitalization
    // read_property(json, tokens + start_index + 30, buffer);

    stock->circulation_market_capitalization = 0;// atof(buffer);

    //trade_num
    // read_property(json, tokens + start_index + 32, buffer);

    stock->trade_num = 0;// atoi(buffer);
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

BOOL next_stock_item(finreader_t *reader)
{
    if(reader == NULL) return FALSE;

    if(reader->cur_index >= reader->size)
    {   
        // if the query out of the boundary, then return last result
        return FALSE;
    }

    stock_t *data_ptr = (stock_t*)(reader->addr + sizeof(meta_t));

    reader->data = data_ptr + (reader->cur_index);

    reader->cur_index += 1;

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