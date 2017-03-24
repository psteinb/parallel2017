PANDOC ?= pandoc

# Pandoc filters.
FILTERS = $(wildcard tools/filters/*.py)

all : index.htm

tools/filters/linkTable : tools/filters/linkTable.hs
	ghc $<

index.htm : slides.md #links.md tools/filters/linkTable
	${PANDOC} -s --no-highlight --highlight-style=espresso --template=pandoc-revealjs.template -t revealjs -o $@ -V transition=slide --section-divs --filter tools/filters/columnfilter.py $< #links.md

# links.md : slides.md
# 	${PANDOC} -t json $< | ./tools/filters/dump_links.py > $@

clean :
	rm -f links.md index.htm?
