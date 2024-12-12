import PyPDF2

#获取 PDF 信息
pdfFile = open('./manual.pdf', 'rb')
pdfObj = PyPDF2.PdfReader(pdfFile)
page_count = len(pdfObj.pages)
text = ""

#提取文本
for p in range(0, page_count):
    page = pdfObj.pages[p]
    text += page.extract_text()
    
with open('./manual.txt', 'w', encoding='utf-8') as file:
    file.write(text)