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

def zng(paragraph):
    for sent in re.findall(u'[^!?。\.\!\?\；]+[!?。\.\!\?\；]?', paragraph, flags=re.U):
        yield sent

def _find_split(lines):
  if len(lines[0])>=15:
    return 0
  if (len(lines[0])+len(lines[1]))>15 and ((len(lines)>2 and len(lines[2])> 15) or (len(lines)>3 and (len(lines[2])+len(lines[3]))>15)):
    return 1
  return -1
def  cleanData(inputPath, outputPath, minLen=30):
  outp=open(outputPath,"w")
  total =0
  skip =0
  toDel1=re.compile(r"⓪|①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|⑴|⑵|⑶|⑷|⑸|⑹|⑺|⑻|⑼|⑽|⒈|⒉|⒊|⒋|⒌")
  toDel2=re.compile(r"([^123456789]*)[1-9]([^123456789]+)")
  
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
      idx = desc.rfind(r"：")
      if idx >0:
        desc = desc[idx+1:]
      idx = desc.rfind(r": ")
      if idx >0:
        desc = desc[idx+1:]
      desc = toDel1.sub(r"", desc)
      desc = toDel2.sub(r"\1\2", desc)
      if len(desc) < minLen:
        skip +=1
        continue
      lns= list(zng(desc))
      if len(lns)<=1:
        skip +=1
        continue
      if len(lns)==2 and (len(lns[0])<10 or len(lns[1])<10):
        skip +=1
        continue
      if len(lns)==3 and (len(lns[0])<10 or len(lns[1])<10 or len(lns[2])<10):
        skip +=1
        continue
      idx = _find_split(lns)
      if idx ==-1:
        skip +=1
        continue
      a = " ".join(lns[:idx+1])
      b = " ".join(lns[idx+1:])
      outp.write("1\t%s\t%s\n"%(a,b))
      outp.write("0\t%s\t%s\n"%(b,a))
  outp.close()
  print("processed %d, skiped :%d"%(total, skip))
  
  
if __name__ =='__main__':
  fire.Fire()
      
    
      