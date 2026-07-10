CFLAGS = -std=c99 -pedantic -Wall -Wextra -O2
LDFLAGS =

.PHONY: all clean

all: twice

twice: twice.c

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f twice
