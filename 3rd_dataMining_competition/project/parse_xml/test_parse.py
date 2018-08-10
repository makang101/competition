#-*- coding:utf-8 -*-

import xml.sax

dict = {}
class ButterflyHandler(xml.sax.ContentHandler):
	def __init__(self):
		self.filename = ''
		self.path = ''
		self.butterflyname = ''
		self.CurrentData = ''     #可以没有
	def startElement(self, tag, atr):
		self.CurrentData = tag
		if tag == 'annotation':
			print('annotation:')
	def endElement(self, tag):
		if tag == 'filename':
			dict['filename'] = self.filename
			#print('filename:',self.filename)
		elif tag == 'path':
			dict['path'] = self.path
			#print('path:',self.path)
		elif tag == 'name':
			dict['name'] = self.butterflyname
	def characters(self, content):
		if self.CurrentData == 'filename':
			self.filename = content
		elif self.CurrentData == 'path':
			self.path = content
		elif self.CurrentData == 'name':
			self.butterflyname = content

if __name__ == '__main__':
	parser = xml.sax.make_parser()
	parser.setFeature(xml.sax.handler.feature_namespaces,0)
	handler = ButterflyHandler()
	print(handler)
	parser.setContentHandler(handler)
	parser.parse('IMG_000001.xml')
	for key in dict:
		print(key ,dict[key])

print('total:',len(dict))