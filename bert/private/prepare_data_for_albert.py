#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: prepare_data_for_albert.py
# Project: private
# Author: koth (Koth Chen)
# -----
# Last Modified: 2019-09-26 4:52:03
# Modified By: koth (nobody@verycool.com)
# -----
# Copyright 2020 - 2019

import fire
import json

def _preprocess_line(desc):
  seeSp =False
  ret=""
  for i in range(len(desc)):
    if  desc[i].isspace():
      if not seeSp:
        ret+=" "
        seeSp = True
    else:
      seeSp = False
      ret+=desc[i]
  return ret
    
def  cleanData(inputPath, outputPath, minLen=80):
  outp=open(outputPath,"w")
  total =0
  skip =0
  with open(inputPath,"r", encoding="utf8") as inp:
    for line in inp:
      line=line.strip()
      total +=1
      if total % 10000 == 0:
        print("processed %d, skiped :%d"%(total, skip))
      if not line:
        skip +=1
        continue
      obj=json.loads(line)
      if 'des' not in obj:
        skip +=1
        continue
      desc= obj['des']
      if len(desc) < minLen:
        skip +=1
        continue
      desc = desc.replace("<br/>","\t")
      desc = desc.replace("\r","\t")
      desc = desc.replace("\n","\t")
      desc = desc.replace("\t\t","\t")
      desc = _preprocess_line(desc)
      if len(desc) < minLen:
        skip +=1
        continue
      outp.write("%s\n"%(desc))
  outp.close()
  print("processed %d, skiped :%d"%(total, skip))
  
  
if __name__ =='__main__':
  fire.Fire()
      
    
      