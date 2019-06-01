import os
import re
import sys 

os.system("rm /tmp/post.md")
os.system("nano /tmp/post.md")

# html_fn = raw_input("location of the html ")
html_fn = sys.argv[1]

image_names = re.findall(r'src=\"images\/image[0-9]+\.png\"' ,open( html_fn.strip() ).read()  )

image_names = [ ii.split("images/image")[1].split(".png")[0] for ii in image_names ]

lines = open("/tmp/post.md").read().split("\n")
for l in lines:
	if "#post_file" in l:
		fn = l.split("#post_file:")[1].split("#")[0].strip()
		break


lines = open("/tmp/post.md").read().split("\n")
for l in lines:
	if "#short_name" in l:
		short_name = l.split("#short_name:")[1].split("#")[0].strip()
		break


print "fn" , fn
print "short_name" , short_name

lines = open("/tmp/post.md").read().split("#post_start")[1].split("\n")
lines = [l for l in lines if not ">>>>>>  gd2m" in l]
lines = [l for l in lines if not "<!-- Docs to Ma" in l]

new_l = []
ii = 0
for l in lines:
	if "![alt_text]" in l or "![drawing]" in l:
		l2 = "![]({{ site.baseurl }}/assets/images/posts/"+short_name+"/image"+image_names[ ii] +".png?style=centerme)"
		ii += 1
		new_l.append(l2)
	else:
		new_l.append(l)

lines = new_l

lines = "\n".join(lines)

lines =  lines.replace("\n\n{: sty" , "\n{: sty")

lines = lines.replace("\n\n---\n\nlay" , "---\n\nlay")

open("_posts/"+ fn +".md",'w').write(lines)

