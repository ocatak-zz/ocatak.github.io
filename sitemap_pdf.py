import glob
from datetime import datetime

fout = open("mysitemap.xml","w")
for fname in glob.glob('files/*.pdf'):
    fout.write("<url>\n    <loc>http://ocatak.github.io/" + fname + "</loc><lastmod>" + datetime.now().strftime("%Y-%m-%d") + "T00:00:00+00:00</lastmod>\n</url>\n" )
fout.close()