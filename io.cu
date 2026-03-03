#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "io.h"

// XXX: Should be moved to a struct with FILE*

// Legacy function to write a single byte to a file, with error handling. 
// Not used 
void put_byte(FILE *fp, int byte)
{
  int status = fputc(byte, fp);

  if (status == EOF)
  {
    fprintf(stderr, "Error writing byte\n");
    exit(EXIT_FAILURE);
  }
}

// Writes a single byte to the internal buffer of the entropy context.
// Used by put_bits(), flush_bits(), and all write_* header functions
// to accumulate encoded data before flushing to disk via flush_frame_to_file().
void put_byte_buf(struct entropy_ctx *c, uint8_t b)
{
  if (c->buf_pos >= c->buf_capacity)  // As long as the buffer has space, we can write to it. If we exceed capacity, it's a critical error.
  {
    fprintf(stderr, "Output buffer overflow\n");
    exit(EXIT_FAILURE);
  }

  c->buf[c->buf_pos++] = b; // Write the byte to the buffer and advance the position.
}

/**
 * Flushes the internal buffer of the entropy context to the file. 
 * This is called after encoding a frame to write all accumulated data to disk at once, 
 * improving performance by reducing the number of write operations.
 */ 
void flush_frame_to_file(struct entropy_ctx *c)
{
  size_t written = fwrite(c->buf, 1, c->buf_pos, c->fp);

  if (written != c->buf_pos)
  {
    fprintf(stderr, "Error flushing frame buffer to file\n");
    exit(EXIT_FAILURE);
  }

  c->buf_pos = 0;
}

// Legacy function to write multiple bytes to a file.
void put_bytes(FILE *fp, const void* data, unsigned int len)
{
  size_t n = fwrite(data, 1, (size_t) len, fp);

  if(n != (size_t) len)
  {
    fprintf(stderr, "Error writing bytes\n");
    exit(EXIT_FAILURE);
  }
}

// Legacy function to read a single byte from a file.
uint8_t get_byte(FILE *fp)
{
  int status = fgetc(fp);

  if (status == EOF)
  {
    fprintf(stderr, "End of file.\n");
    exit(EXIT_FAILURE);
  }

  return (uint8_t) status;
}

// Legacy function to read multiple bytes from a file.
int read_bytes(FILE *fp, void *data, unsigned int sz)
{
  size_t status = fread(data, 1, (size_t) sz, fp);

  if ((int) status == EOF)
  {
    fprintf(stderr, "End of file.\n");
    exit(EXIT_FAILURE);
  }
  else if (status != (size_t) sz)
  {
    fprintf(stderr, "Error reading bytes\n");
    exit(EXIT_FAILURE);
  }

  return (int) status;
}

/**
 * Adds a bit to the bitBuffer. A call to bit_flush() is needed
 * in order to write any remainding bits in the buffer before
 * writing using another function.
 */
void put_bits(struct entropy_ctx *c, uint16_t bits, uint8_t n)
{
  assert(n <= 24  && "Error writing bit");

  if(n == 0) { return; }

  c->bit_buffer <<= n;
  c->bit_buffer |= bits & ((1 << n) - 1);
  c->bit_buffer_width += n;

  while(c->bit_buffer_width >= 8)
  {
    uint8_t b = (uint8_t)(c->bit_buffer >> (c->bit_buffer_width - 8));

    put_byte_buf(c, b);

    if(b == 0xff) { put_byte_buf(c, 0); }

    c->bit_buffer_width -= 8;
  }
}

/**
 * Reads n bits from the bitBuffer. A call to flush_bits() is needed
 * in order to write any remaining bits in the buffer before
 * writing using another function.
 */
uint16_t get_bits(struct entropy_ctx *c, uint8_t n)
{
  uint16_t ret = 0;

  while(c->bit_buffer_width < n)
  {
    uint8_t b = get_byte(c->fp);
    if (b == 0xff) { get_byte(c->fp); } /* Discard stuffed byte */

    c->bit_buffer <<= 8;
    c->bit_buffer |= b;
    c->bit_buffer_width += 8;
  }

  ret = c->bit_buffer >> (c->bit_buffer_width - n);
  c->bit_buffer_width -= n;

  /* Clear grabbed bits */
  c->bit_buffer &= (1 << c->bit_buffer_width) - 1;

  return ret;
}

/**
 * Flushes the bitBuffer by writing zeroes to fill a full byte
 */
void flush_bits(struct entropy_ctx *c)
{
  if(c->bit_buffer > 0)
  {
    uint8_t b = c->bit_buffer << (8 - c->bit_buffer_width);
    put_byte_buf(c, b);

    if(b == 0xff) { put_byte_buf(c, 0); }
  }

  c->bit_buffer = 0;
  c->bit_buffer_width = 0;
}
