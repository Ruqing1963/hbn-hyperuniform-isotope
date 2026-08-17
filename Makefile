.PHONY: all sequences structure transport paper clean test

N_SITES ?= 200000

all: sequences structure transport paper

sequences:
	cd code && python generate_sequence.py --all --n-sites $(N_SITES)

structure: sequences
	cd code && python structure_factor_check.py

transport: sequences
	cd code && python transport.py

paper:	## real REVTeX build; needs texlive-publishers
	cd paper && pdflatex -interaction=nonstopmode paper.tex \
	  && bibtex paper || true \
	  && pdflatex -interaction=nonstopmode paper.tex \
	  && pdflatex -interaction=nonstopmode paper.tex

test:
	pytest -q tests

clean:
	rm -f data/processed/*.npz data/processed/*.csv figures/*.pdf
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg
	rm -f paper/paper.pdf paper/paper_preview.pdf paper/paper_zh.pdf

preview:  ## readable PDF without REVTeX -- NOT for submission
	cd paper && python3 _mkpreview.py \
	  && pdflatex -interaction=nonstopmode preview.tex \
	  && pdflatex -interaction=nonstopmode preview.tex \
	  && mv preview.pdf paper_preview.pdf \
	  && rm -f preview.tex preview.aux preview.log preview.out

zh:  ## Chinese reading version (xelatex + Noto CJK); not for submission
	cd paper && xelatex -interaction=nonstopmode paper_zh.tex \
	  && xelatex -interaction=nonstopmode paper_zh.tex \
	  && rm -f paper_zh.aux paper_zh.log paper_zh.out
