PANDOC ?= pandoc

all : heisedev.pdf

%.pdf : %.md
	$(PANDOC) -o $@ $<

clean :
	@rm -fv *.pdf *.log
