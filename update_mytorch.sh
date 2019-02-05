#!/usr/bin/env bash
rm -rf mytorch
git clone https://github.com/geraltofrivia/mytorch.git -q
cd mytorch
chmod +x setup.sh
./setup.sh
cd ..