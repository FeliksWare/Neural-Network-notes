CFLAGS = -std=c99 -pedantic -Wall -Wextra -O2 -ggdb
LDFLAGS = -lm

.PHONY: all clean

all: twice gates xor xor_nn

twice: twice.c
gates: gates.c
xor: xor.c
xor_nn: xor_nn.c

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f twice gates xor xor_nn
