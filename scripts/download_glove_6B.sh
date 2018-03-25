#!/usr/bin/env bash

FILE=glove.6B.zip
URL=http://nlp.stanford.edu/data/glove.6B.zip
CHECKSUM=056ea991adb4740ac6bf1b6d9b50408b

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../models/GloVe/" && pwd )"
cd $DIR

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading GloVe models..."

wget $URL -O $FILE

echo "Unzipping..."

unzip $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
