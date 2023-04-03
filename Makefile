FILENAME := doc
all : doc
 
doc : $(FILENAME).md 
	pandoc --filter pandoc-plot --listings --pdf-engine=xelatex --variable papersize=a4paper -C -s $(FILENAME).md -o $(FILENAME).pdf
	open $(FILENAME).pdf
clean :
	rm $(FILENAME).pdf
