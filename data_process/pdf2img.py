import pymupdf  

fname = "data/NVIDIA-10Q-20242905.pdf"
doc = pymupdf.open(fname)  

zoom_x = 2.5  
zoom_y = 2.5  
mat = pymupdf.Matrix(zoom_x, zoom_y)  

for page in doc:  
    pix = page.get_pixmap(matrix=mat)  
    pix.save("data/NVIDIA-10Q-20242905/page-%i.png" % page.number)  