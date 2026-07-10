CFLAGS = -std=c99 -pedantic -Wall -Wextra -O2
LDFLAGS = -lm

.PHONY: all clean

all: twice gates

twice: twice.c
gates: gates.c

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f twice gates
