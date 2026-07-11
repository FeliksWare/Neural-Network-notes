CFLAGS = -std=c99 -pedantic -Wall -Wextra -O2
LDFLAGS = -lm

.PHONY: all clean

all: twice gates xor

twice: twice.c
gates: gates.c
xor: xor.c

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f twice gates xor
