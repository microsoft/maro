#include "combine.h"

int main(int argc, char *argv[])
{
    for(int i=0;i<argc;i++)
    {
        printf("%s.\n", argv[i]);
    }

    return 0;
}

void process(char *root, char *item_names[])
{

}


void init_writer(char *path, combine_writer_t *writer, int64_t start_time, int64_t end_time, int16_t item_number)
{
    writer->file = fopen(path, "wb+");

    combine_header_t t = {item_number, sizeof(stock_t), start_time, end_time};

    writer->header = t;

    fwrite(&t, sizeof(t), 1, writer->file);
}

void release_writer(combine_writer_t *writer)
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

void new_row(combine_writer_t *writer, int64_t time)
{
    if(writer == NULL)
    {
        return;
    }

    combine_row_meta_t row_meta = {0, time};

    fwrite(&row_meta, sizeof(combine_row_meta_t), 1, writer->file);
}

void update_item_number(combine_writer_t *writer, int16_t item_number)
{
    if(writer == NULL)
    {
        return;
    }

    fseek(writer->file, writer->header.item_length * item_number, SEEK_END);

    fwrite(&item_number, sizeof(int16_t), 1, writer->file);
}

void add_stock(combine_writer_t *writer, stock_t *stock)
{
    if(writer == NULL)
    {
        return;
    }

    fwrite(stock, sizeof(stock_t), 1, writer->file);
}