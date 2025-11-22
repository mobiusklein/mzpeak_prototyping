#include <R.h>
#include <Rinternals.h>
#include <R_ext/Connections.h>

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>

typedef struct connection_segment {
  char* source_name;
  char* member_name;
  FILE* stream;
  size_t offset;
  size_t position;
  size_t size;
} ConnectionSegment;


static int create_connection_segment(ConnectionSegment* con,
                                     char* source_file,
                                     char* member_name,
                                     size_t offset,
                                     size_t size) {
  con->source_name = source_file;
  con->member_name = member_name;
  con->offset = offset;
  con->position = 0;
  con->size = size;
  con->stream = NULL;
  return 0;
}

static int destroy_connection_segment(ConnectionSegment* con) {
  if (con != NULL) {
    fclose(con->stream);
    free(con->source_name);
    free(con->member_name);
    free(con);
  }
  return 0;
}

static size_t seg_read(void *ptr, size_t size, size_t nitems,
                       Rconnection con) {
  ConnectionSegment* myconn = con->private;
  size_t to_read = size * nitems;
  fseeko64(myconn->stream, myconn->offset + myconn->position, SEEK_SET);
  if ((to_read + myconn->position) > myconn->size) {
    size_t remaining_bytes = (myconn->size - myconn->position);
    size_t remaining_items = remaining_bytes / size;
    printf("Had to reduce to %zu items for a total of %zu bytes\n", remaining_items, remaining_bytes);
    fread(ptr, size, remaining_items, myconn->stream);
    myconn->position += remaining_items * size;
    return remaining_items;
  } else {
    fread(ptr, size, nitems, myconn->stream);
    myconn->position += to_read;
    return nitems;
  }
}


static int read_zip_header_for_size(ConnectionSegment* myconn, size_t* bytes_read_out) {
  char magic[4] = {0, 0, 0, 0};
  size_t bytes_read = fread(&magic, 1, 4, myconn->stream);

  char expected_magic[4] = {0x50, 0x4B, 0x03, 0x04};
  for(int i = 0; i < 4; i++) {
    if (magic[i] != expected_magic[i]) {
      return 1;
    }
  }

  // Read the "minimum version" bit flag
  char version[2] = {0, 0};
  bytes_read += fread(&version, 1, 2, myconn->stream);

  // Read the "general purpose" bit flag
  bytes_read += fread(&version, 1, 2, myconn->stream);

  short compression_method = 0;
  // Read the "compression" bit flag
  bytes_read += fread(&compression_method, 1, 2, myconn->stream);
  if (compression_method != 0) {
    return 2;
  }

  // Read the "last modified time" bytes
  bytes_read += fread(&version, 1, 2, myconn->stream);

  // Read the "last modified date" bytes
  bytes_read += fread(&version, 1, 2, myconn->stream);

  // Read the "CRC" bytes
  bytes_read += fread(&magic, 1, 4, myconn->stream);

  // Read the "compressed size" bytes
  bytes_read += fread(&magic, 1, 4, myconn->stream);

  // Read the "uncompressed size" bytes
  bytes_read += fread(&magic, 1, 4, myconn->stream);

  unsigned short filename_size = 0;
  bytes_read += fread(&filename_size, 2, 1, myconn->stream) * 2;

  unsigned short extra_size = 0;
  bytes_read += fread(&extra_size, 2, 1, myconn->stream) * 2;

  bytes_read += filename_size;
  bytes_read += extra_size;

  *bytes_read_out = bytes_read;
  return 0;
}


static Rboolean seg_open(Rconnection con) {
  ConnectionSegment* myconn = (ConnectionSegment*) con->private;
  myconn->stream = fopen(myconn->source_name, "rb");
  printf("Seeking to %zu within %s\n", myconn->offset, myconn->source_name);
  fseeko64(myconn->stream, myconn->offset, SEEK_SET);
  myconn->position = 0;

  size_t header_bytes = 0;
  int rc = read_zip_header_for_size(myconn, &header_bytes);
  if (rc == 1) {
    error("Failed to verify ZIP local header magic");
  } else if (rc == 2) {
    error("Compression detected, only uncompressed ZIP archives are supported");
  }

  myconn->offset += header_bytes;
  fseeko64(myconn->stream, myconn->offset, SEEK_SET);
  myconn->position = 0;
  con->isopen = TRUE;
  con->canread = TRUE;
  con->canwrite = FALSE;
  con->canseek = TRUE;
  con->text = FALSE;  // Binary mode
  return TRUE;
}

static void seg_close(Rconnection con) {
  ConnectionSegment* myconn = (ConnectionSegment*) con->private;
  con->isopen = FALSE;
  destroy_connection_segment(myconn);
  con->private = NULL;
}

static double seg_seek(Rconnection con, double where, int origin, int rw) {
  ConnectionSegment* myconn = (ConnectionSegment*) con->private;
  printf("Seeking %f from origin %d\n", where, origin);
  if (isnan(where)) {
    return myconn->position;
  }
  if (origin == 1) {
    if ((size_t)where > myconn->size) {
      printf("Asked to seek from start beyond %zu, clamping to size", myconn->size);
      where = myconn->size;
    }
    size_t newpos = myconn->offset + (size_t)where;
    fseeko64(myconn->stream, newpos, SEEK_SET);
    myconn->position = (size_t)where;
  }
  else if (origin == 3) {
    if (fabs(where) > myconn->size) {
      printf("Asked to seek from end beyond %zu, clamping to size", myconn->size);
      where = myconn->size;
    }
    size_t newpos = (myconn->offset + myconn->size) - (size_t)where;
    fseeko64(myconn->stream, newpos, SEEK_SET);
    myconn->position = myconn->size + where;
  }
  else if (origin == 2) {
    if ((where + myconn->position) > myconn->size) {
      printf("Asked to seek from current beyond %zu, clamping to size", myconn->size);
      where = myconn->size - myconn->position;
    }
    fseeko64(myconn->stream, myconn->offset + myconn->position + where, SEEK_SET);
    myconn->position += where;
  } else {
    error("Could not interpret seek origin");
  }
  printf("Stream position is %zu post-seek, virtual position is %zu\n", ftello64(myconn->stream), myconn->position);
  return (double) myconn->position;
}

SEXP R_make_connection_segment(SEXP source_file, SEXP member_name, SEXP offset, SEXP size) {
  Rconnection con;
  SEXP rc = PROTECT(
    R_new_custom_connection(
      translateCharUTF8(asChar(source_file)),
      "rb",
      "connection_segment",
      &con
    )
  );
  const char* source_file_c = translateCharUTF8(asChar(source_file));
  const char* member_name_c = translateCharUTF8(asChar(member_name));

  size_t zsource = strlen(source_file_c) + 1;
  char* source_file_owned = malloc(sizeof(char) * zsource);
  strcpy(source_file_owned, source_file_c);

  size_t zmember = strlen(member_name_c) + 1;
  char* member_name_owned = malloc(sizeof(char) * zmember);
  strcpy(member_name_owned, member_name_c);

  size_t offset_c = asReal(offset);
  size_t size_c = asReal(size);

  ConnectionSegment* handle = malloc(sizeof(ConnectionSegment));
  int ec = create_connection_segment(
    handle,
    source_file_owned,
    member_name_owned,
    offset_c,
    size_c
  );

  if (ec != 0) {
    error("Failed to allocate connection segment");
  }
  char* description = malloc(255);
  sprintf(description, "a connection segment starting at %zu with size %zu",
          handle->offset, handle->size);

  con->private = handle;
  con->canseek = TRUE;
  con->canread = TRUE;
  con->canwrite = FALSE;
  con->isopen = FALSE;
  con->blocking = TRUE;
  con->text = FALSE;
  con->UTF8out = FALSE;
  con->open = seg_open;
  con->close = seg_close;
  con->seek = seg_seek;
  con->read = seg_read;
  con->description = description;
  UNPROTECT(1);
  return rc;
}

static const R_CallMethodDef CallEntries[] = {
  {"_make_connection_segment", (DL_FUNC) &R_make_connection_segment, 4},
  {NULL, NULL, 0}
};


void R_init_mzpeak(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
