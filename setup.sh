#!/usr/bin/env bash

/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install python3

python3 -m venv tradebotvenv
source tradebotvenv/bin/activate
pip3 install -r requirements.txt
