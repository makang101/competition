#-*- coding:utf-8 -*-

import xml.sax
import os
dict_count = {}
dict = {}
class ButterflyHandler(xml.sax.ContentHandler):
	def __init__(self):
		# self.filename = ''
		# self.path = ''
		self.butterflyname = ''
		self.CurrentData = ''     #可以没有
	def startElement(self, tag, atr):
		self.CurrentData = tag
		
	def endElement(self, tag):
		if tag == 'name':
			dict['butterflyname'] = self.butterflyname			

	def characters(self, content):
		if self.CurrentData == 'name':
			self.butterflyname = content

if __name__ == '__main__':
	filepath = input('enter your analy file path:')
	filenames = os.listdir(filepath)
	print(filenames)
	for f in filenames:
		parser = xml.sax.make_parser()
		parser.setFeature(xml.sax.handler.feature_namespaces,0)
		handler = ButterflyHandler()
		parser.setContentHandler(handler)
		parser.parse(f)
		#print(dict['butterflyname'])
		if dict['butterflyname'] in dict_count:
			dict_count[dict['butterflyname']] += 1
		else:
			dict_count[dict['butterflyname']] = 1
	
	for k,v in dict_count.items():
		print(k,v)