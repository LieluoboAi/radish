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
import re
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
    
def  cleanData(inputPath, outputPath, minLen=40):
  outp=open(outputPath,"w")
  total =0
  skip =0
  p = re.compile("。|，|？")
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
      desc = desc.replace("<br/>"," ")
      desc = desc.replace("\r"," ")
      desc = desc.replace("\n"," ")
      desc = desc.replace("\t"," ")
      desc = _preprocess_line(desc)
      if len(desc) < minLen:
        skip +=1
        continue
      lns= p.split(desc)
      if len(lns)<=1:
        skip +=1
        continue
      if len(lns)==2 and (len(lns[0])<25 or len(lns[1])<25):
        skip +=1
        continue
      if len(lns)==3 and (len(lns[0])<25 or len(lns[1])<25 or len(lns[2])<25):
        skip +=1
        continue
      outp.write("%s\n"%("\t".join(lns)))
  outp.close()
  print("processed %d, skiped :%d"%(total, skip))
  
  
if __name__ =='__main__':
  fire.Fire()
      
    
      