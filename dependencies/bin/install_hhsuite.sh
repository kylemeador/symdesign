#!/bin/bash

cd $SymDesign/dependencies
git clone https://github.com/soedinglab/hh-suite.git
mkdir -p hh-suite/build && cd hh-suite/build
cmake -DCMAKE_INSTALL_PREFIX=. ..
make -j 4 && make install

# Need to match
# <tr class="even"><td class="indexcolicon"><a href="UniRef30_2022_02_hhsuite.tar.gz"><img src="/~compbiol/uniclust/theme/icons/archive.png" alt="[   ]" /></a></td><td class="indexcolname"><a href="UniRef30_2022_02_hhsuite.tar.gz">UniRef30_2022_02_hhsuite.tar.gz</a></td><td class="indexcolsize"> 58G</td>
wget http://wwwuser.gwdg.de/~compbiol/uniclust/current_release/`wget -O - http://wwwuser.gwdg.de/~compbiol/uniclust/current_release/ | grep -Po 'indexcolname"><a\ href="\K[^"]*_hhsuite.tar.gz'`
tar xzf `wget -O - http://wwwuser.gwdg.de/~compbiol/uniclust/current_release/ | grep -Po 'indexcolname"><a\ href="\K[^"]*_hhsuite.tar.gz'` -C hh-suite/databases/
echo export PATH="$SymDesign/dependencies/hh-suite/build/bin:$SymDesign/dependencies/hh-suite/build/scripts:$PATH" >> ~/.profile
