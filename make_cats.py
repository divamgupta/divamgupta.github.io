# a hack coz github pages is soo buggy!!

import os

temp = """---
layout: categorie
title: %s
permalink: /category/%s
---"""

cats = os.listdir("./_site/category")
for c in cats:
	t = temp%(c,c)
	open("ccc/"+c+".md",'w').write(t)