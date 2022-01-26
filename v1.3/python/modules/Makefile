SHELL=/bin/bash

all:

clean:
	/bin/rm -f  *.o
	/bin/rm -f  *.so
	/bin/rm -f  *.pyc *.pyo
	/bin/rm -rf __pycache__

cleanup clobber: clean

cleanupall distclean: cleanup

.PHONY: all clean cleanup clobber cleanupall
