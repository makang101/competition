
# coding: utf-8

# In[1]:


import xml.etree.ElementTree as ET
import os


# In[2]:


def xml_parse(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    object = root.find('object')
    name = object.find('name').text
    return name


# In[4]:


def get_namemapnum(input_dir):
    #input_dir = 'C:\Users\Administrator\work\pase_excel\第三届中国数据挖掘大赛-蝴蝶训练集\Annotations'
    filenamelist = os.listdir(input_dir)
    #print(filepathlist)
    dict = {}
    for filename in filenamelist:
        filepath = input_dir + '\\' + filename
        name = xml_parse(filepath)
        if name in dict:
            dict[name] += 1
        else:
            dict[name] = 1
#     for name, n in dict.items():
#         print(name,':',n)
    return dict
    

