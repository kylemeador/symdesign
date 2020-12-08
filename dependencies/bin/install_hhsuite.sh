#!/bin/bash

cd $SymDesign/dependencies
git clone https://github.com/soedinglab/hh-suite.git
mkdir -p hh-suite/build && cd hh-suite/build
cmake -DCMAKE_INSTALL_PREFIX=. ..
make -j 4 && make install

wget http://wwwuser.gwdg.de/~compbiol/uniclust/current_release/`wget -O - http://wwwuser.gwdg.de/~compbiol/uniclust/current_release/ | grep -Po 'indexcolname"><a\ href="\K[^"]*_hhsuite.tar.gz'`
tar xzf `wget -O - http://wwwuser.gwdg.de/~compbiol/uniclust/current_release/ | grep -Po 'indexcolname"><a\ href="\K[^"]*_hhsuite.tar.gz'` -C hh-suite/databases/
echo export PATH="$SymDesign/dependencies/hh-suite/build/bin:$SymDesign/dependencies/hh-suite/build/scripts:$PATH" >> ~/.profile
