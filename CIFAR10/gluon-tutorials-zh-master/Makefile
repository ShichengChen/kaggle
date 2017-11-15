all: html

build/%.ipynb: %.md build/build.yml utils.py
	@mkdir -p $(@D)
	cd $(@D); python ../md2ipynb.py ../../$< ../../$@

build/%.md: %.md
	@mkdir -p $(@D)
	@cp $< $@

MARKDOWN = $(wildcard chapter_preface/*.md */index.md)
NOTEBOOK = $(filter-out $(MARKDOWN), $(wildcard chapter*/*.md))

OBJ = $(patsubst %.md, build/%.md, $(MARKDOWN)) \
	$(patsubst %.md, build/%.ipynb, $(NOTEBOOK))

ORIGN_DEPS = $(wildcard img/* data/*) environment.yml utils.py README.md
DEPS = $(patsubst %, build/%, $(ORIGN_DEPS))

PKG = build/_build/html/gluon_tutorials_zh.tar.gz build/_build/html/gluon_tutorials_zh.zip

pkg: $(PKG)

build/_build/html/gluon_tutorials_zh.zip: $(OBJ) $(DEPS)
	cd build; zip -r $(patsubst build/%, %, $@ $(DEPS)) chapter*

build/_build/html/gluon_tutorials_zh.tar.gz: $(OBJ) $(DEPS)
	cd build; tar -zcvf $(patsubst build/%, %, $@ $(DEPS)) chapter*

build/%: %
	@mkdir -p $(@D)
	@cp -r $< $@

html: $(DEPS) $(OBJ)
	make -C build html
	bash build/htaccess.sh build/_build/html/

latex: $(DEPS) $(OBJ)
	make -C build latex

clean:
	rm -rf build/chapter* $(DEPS) $(PKG)
